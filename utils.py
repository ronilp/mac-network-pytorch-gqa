# File: utils.py
# Author: Ronil Pancholia
# Date: 4/22/19
# Time: 7:57 PM
import pickle
import sys

import numpy as np

id2word = []
embedding_weights = None

def get_or_load_embeddings():
    global embedding_weights, id2word

    if embedding_weights is not None:
        return embedding_weights

    dataset_type = sys.argv[1]
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    id2word = set(dic['word_dic'].keys())
    id2word.update(set(dic['answer_dic'].keys()))

    word2id = {word: id for id, word in enumerate(id2word)}

    embed_size = 300
    vocab_size = len(id2word)
    sd = 1 / np.sqrt(embed_size)
    embedding_weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    embedding_weights = embedding_weights.astype(np.float32)

    with open("data/glove.6B.300d.txt", encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = word2id.get(word, None)
            if id is not None:
                embedding_weights[id] = np.array(line[1:], dtype=np.float32)

    return embedding_weights
