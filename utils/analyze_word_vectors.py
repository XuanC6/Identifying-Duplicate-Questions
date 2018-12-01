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
data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/DupQues/data/"
word2vec_model_path = data_dir + "word2vec.model"

# load model
model = Word2Vec.load(word2vec_model_path)
word_vectors = model.wv

# num of words
print("num of words: %d" % (len(word_vectors.vocab)))

# t-SNE
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    count = 0
    for word in model.wv.vocab:
        if count<150:
            tokens.append(model[word])
            labels.append(word)
            count += 1
        else:
            break

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=3000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(20, 15))
    frame = plt.gca()
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    frame.axes.set_yticklabels([]) #Hide ticks
    frame.axes.set_xticklabels([]) #Hide ticks
    plt.show()
    plt.savefig("word_embedding_visual.png")

tsne_plot(model)