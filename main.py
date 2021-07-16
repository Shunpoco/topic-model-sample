import MeCab
import pandas as pd
from pathlib import Path, PosixPath
from collections import Counter, OrderedDict
from functools import reduce
import re
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from typing import List

PATH = './data/text'

def print_text(texts: List[str]):
    for i, t in enumerate(texts):
        print(i, t.strip())

def ls(path: str) -> List[PosixPath]:
    data_dir = Path(path)
    files = sorted([f for f in data_dir.glob('*/*[0-9].txt')])
    print(f'{len(files)} files')

    return files

def count_file(file_path: str) -> Counter:
    """テキストデータの前処理
    トピックモデルを適用するには、文書をカウントデータに変換する必要がある。
    この関数では、bag of words (bow) を使用する。
    bow は、単語の出現回数を成分に持つベクトルで表現される。

    1. 各テキストファイルに含まれる単語をカウント
    2. 全テキストファイルに含まれる単語の出現回数を得る
    3. ストップワードを除く
    4. 単語の出現回数の上位5000からなるbow表現を得る
    
    inputs:
    - file_path(str): ファイルパス

    returns:
    - Counter: word counter
    """

    with file_path.open() as f:
        lines = f.readlines()
    doc = ''.join(lines[2:])

    chasen = MeCab.Tagger('-Ochasen')
    parsed = chasen.parse(doc)

    counter = Counter()

    for splited in parsed.split('\n'):
        if splited == 'EOS':
            break

        s: List[str] = splited.split('\t')

        word = s[2]
        part_of_speech = s[3]
        if '名詞' in part_of_speech:
            counter[word] += 1

    return counter

def remove_stops(counter: Counter) -> Counter:
    for word in set(counter.elements()):
        if re.search(r'^[0-9a-zA-Z.\-ー/(),]$', word):
            del counter[word]

    return counter

def bag_of_words(word_counter: Counter, counts: OrderedDict, n: int=5000) -> pd.DataFrame:
    columns = [x[0] for x in word_counter.most_common(n)]

    data = dict()
    for column in columns:
        data[column] = [count[column] for count in counts.values()]

    index = list(counts.keys())
    df = pd.DataFrame(data, index)

    print(df.iloc[:5, :10])
    print(df.shape)

    return df


def topic_model(X: np.ndarray, k=10) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(n_components=k, random_state=213)
    lda.fit(X)

    return lda

def print_top_words(model: LatentDirichletAllocation, feature_names: str, n_top_words: int):
    """各トピックの確率の高い単語を上からn個表示する
    """
    for topic_idx, topic in enumerate(model.components_):
        message = f'トピック {topic_idx}: '
        message += ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])

        print(message)

def print_topic_docs(doc_topic_distr: np.ndarray, topic_index: int, df: pd.DataFrame, length=10):
    """トピックごとの文書のファイル名とタイトルを表示する。"""
    indices = np.argsort(doc_topic_distr[:, topic_index])[::-1][:length]
    docs = df.index[indices]

    for i, file_path in enumerate(docs):
        file_path
        with open(file_path, 'r') as f:
            text = f.readlines()
        title = text[2].strip()

        print(f'{i}: {title}')

if __name__ == '__main__':
    # with open('./data/text/dokujo-tsushin/dokujo-tsushin-4778030.txt', 'r') as f:
    #     text = f.readlines()

    # print_text(text[:10])

    files = ls(PATH)
    print(files)
    counts = OrderedDict()

    for file in files:
        counts[str(file)] = count_file(file)
    
    total_counts = reduce(lambda x, y: x + y, counts.values(), Counter())

    num_types = len(set(total_counts.elements()))
    print(f'Kinds of words: {num_types}')
    print(f'Number of words: {len(list(total_counts.elements()))}')

    print(f'Top 20th: {total_counts.most_common(20)}')

    total_counts_removed = remove_stops(total_counts)

    num_types_removed = len(set(total_counts_removed))
    print(f'Kinds of words after remove stopwords: {num_types_removed}')
    print(f'{num_types - num_types_removed}')

    df = bag_of_words(total_counts_removed, counts)

    lda = topic_model(df.values)

    print_top_words(lda, df.columns, 20)

    doc_topic_distr = lda.transform(df.values)

    print('-'*20)
    print_topic_docs(doc_topic_distr, 1, df)
    print('-'*20)
    print_topic_docs(doc_topic_distr, 2, df)
    print('-'*20)
    print_topic_docs(doc_topic_distr, 3, df)
