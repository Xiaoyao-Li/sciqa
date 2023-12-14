#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:30:29 2018
Modified on Wed Dec 14 2023
@author: manojacharya, Puhao Li
"""
import numpy as np
import json


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(val) for val in  vals[1:]]
        word2emb[word] = np.array(vals)
    for word in idx2word.keys():
        if word not in word2emb:
            print('word %s not in glove embedding' % word)
            continue
        idx = idx2word[word]
        print('word %s in glove embedding' % word)
        weights[idx] = word2emb[word]
    return weights, word2emb


# if __name__ == '__main__':
#     vocab = json.load(open('datasets/assets/scienceqa_vocab.json', 'r'))
#     question_vocab = vocab['question']

#     emb_dim = 300
#     glove_file = './preprocess/assets/glove.6B.%dd.txt' % emb_dim
#     weights, word2emb = create_glove_embedding_init(question_vocab, glove_file)
#     np.save('models/model/assets/glove6b.init_scienceqa_onlyquestion_%dd.npy' % emb_dim, weights)

# if __name__ == '__main__':
#     vocab = json.load(open('datasets/assets/scienceqa_vocab.json', 'r'))
#     question_vocab = vocab['question_and_hint']

#     emb_dim = 300
#     glove_file = './preprocess/assets/glove.6B.%dd.txt' % emb_dim
#     weights, word2emb = create_glove_embedding_init(question_vocab, glove_file)
#     np.save('models/model/assets/glove6b.init_scienceqa_questionandhint_%dd.npy' % emb_dim, weights)
