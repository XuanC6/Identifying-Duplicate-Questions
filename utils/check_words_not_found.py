# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:25:40 2018

@author: CX
"""
import os
import numpy as np

data_dir = os.path.dirname(os.path.dirname(__file__)) + "/baseline/data/"
indices_dir = data_dir + "indices/"
train_dir = data_dir + "train/"
val_dir = data_dir + "valid/"
glove_dir = data_dir + "/glove.6B"


word_index = {}
with open(data_dir+"id_word.txt", "r", encoding='UTF-8') as f:
    for line in f:
        line = line.strip('\n')
        values = line.split('\t')
        word = values[1]
        idx = values[0]
        word_index[word] = idx

# In[glove word embeddings]
embeddings_index = {}
with open(os.path.join(glove_dir, "glove.6B.100d.txt"), "r", encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = np.float32)
        embeddings_index[word] = coefs

print('Found %s word vectors.'%len(embeddings_index))

words_not_found = 0
words_not_found_list = []
#embedding_matrix = np.random.rand(len(word_index)+1, embedding_dim)
#embedding_matrix = (embedding_matrix-0.5)/10
#embedding_matrix[0] = np.zeros(embedding_dim)

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        words_not_found += 1
        words_not_found_list.append(word)
