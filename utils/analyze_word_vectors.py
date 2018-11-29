# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:21:21 2018

@author: CX
"""
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# path
data_dir = os.path.dirname(os.path.dirname(__file__)) + "/DupQues/data/"
word2vec_model_path = data_dir + "word2vec.model"

# load model
model = Word2Vec.load(word2vec_model_path)
word_vectors = model.wv

# num of words
print("num of words: %d" % (len(word_vectors.vocab)))

# t-SNE
wanted_words = []
count = 0
for word in word_vectors.vocab:
    if count<120:
        wanted_words.append(word)
        count += 1
    else:
        break
wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)

X = model[wanted_vocab] # X is an array of word vectors, each vector containing 150 tokens
tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=5000, random_state=23)
Y = tsne_model.fit_transform(X)

#Plot the t-SNE output
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
ax.set_yticklabels([]) #Hide ticks
ax.set_xticklabels([]) #Hide ticks
plt.show()