# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:51:00 2018

@author: CX
"""
import os
import re
import csv
import numpy as np
from collections import defaultdict


# In[path]
data_dir = os.path.dirname(os.path.dirname(__file__)) + "/baseline/data/"
indices_dir = data_dir + "indices/"
train_dir = data_dir + "train/"
val_dir = data_dir + "valid/"
glove_dir = data_dir + "/glove.6B"

if not os.path.exists(indices_dir):
    os.mkdir(indices_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

load = True
word_index = {}
word_counts = defaultdict(lambda: 0)

signs = '?.？。!！ '
split_signs = r'\s*[—…⚡；？！。，、″¨“”：（）《》【】\\!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\s]\s*'


# In[read data]
# word list -> index list
# meanwhile, count words
def word2idx(q: list) -> list:
    q_seq = []

    for word in q:
        word = word.strip("' ").lower()
        if word == '' or word == ' ':
            continue
        if word not in word_index:
            # indexing
            word_index[word] = len(word_index)+1
        
        word_counts[word] += 1
        q_seq.append(word_index[word])

    return q_seq


with open(data_dir+"questions.csv",'r',encoding='UTF-8') as f:
    q_seqs = []
    labels = []
    
    f_csv = csv.reader(f)
    apostrophe = "′‘’´`"
    for line in f_csv:
        # ignore first line
        if f_csv.line_num == 1:
                continue
        # String
        q1 = line[3]
        q2 = line[4]
        for s in apostrophe:
            q1 = q1.replace(s, "'")
            q2 = q2.replace(s, "'")

        # def split(pattern, string, maxsplit=0, flags=0):
        q1_list = re.split(split_signs, q1.strip(signs))    # signs = '?.？。!！ '
        q2_list = re.split(split_signs, q2.strip(signs))

        q_seqs.append(word2idx(q1_list))    # every two line,
        q_seqs.append(word2idx(q2_list))    # put q1 and q2's word list
        labels.append(int(line[5]))


# In[save dicts and parsed data]
index_word = {value:key for key, value in word_index.items()}   # turn word_index into index_word
with open(data_dir+"data_parsed.txt","w",encoding='UTF-8') as f:
    for seq in q_seqs:
        for s in seq:
            f.write(index_word[s] + ' ')
        f.write('\n')
    # \\s, \n

with open(data_dir+"id_word.txt","w",encoding='UTF-8') as f:
    for word, idx in word_index.items():
        f.write(str(idx) + '\t' + word + '\n')


with open(data_dir+"word_counts.txt","w",encoding='UTF-8') as f:
    
    word_counts_list = list(word_counts.items())
    word_counts_list = reversed(sorted(word_counts_list, key = lambda x: x[1]))
    
    for word, counts in word_counts_list:
        f.write(word + '\t' + str(counts) + '\n')

print('Found %s unique tokens.' % len(word_index))


# In[process data]
max_len = 35
training_samples = 380000

def pad_seqs(seqs: list, maxlen: int)->np.array:
    for i in range(len(seqs)):
        length = len(seqs[i])
        if length > maxlen:
            seqs[i] = seqs[i][:maxlen]
        elif length < maxlen:
            seqs[i].extend([0]*(maxlen - length))
    
    return np.asarray(seqs)

lengths = []
for s in q_seqs:
    lengths.append(len(s))

data = pad_seqs(q_seqs, max_len)
labels = np.asarray(labels)
lengths = np.asarray(lengths)


# In[save data]
question_set1 = data[::2]
question_set2 = data[1::2]
lengths_set1 = lengths[::2]
lengths_set2 = lengths[1::2]
assert question_set1.shape[0] == question_set2.shape[0]


if os.path.exists(indices_dir+"indices.txt") and load:
    indices = np.loadtxt(indices_dir+"indices.txt", dtype = np.int32)
else:
    indices = np.arange(question_set1.shape[0])
    np.random.shuffle(indices)
    np.savetxt(indices_dir+"indices.txt", indices, fmt='%d')


question_set1 = question_set1[indices]
question_set2 = question_set2[indices]
labels = labels[indices]
lengths_set1 = lengths_set1[indices]
lengths_set2 = lengths_set2[indices]

data1_name = "data1.txt"
data2_name = "data2.txt"
labels_name = "labels.txt"
length1_name = "length1.txt"
length2_name = "length2.txt"

x_train_set1 = question_set1[:training_samples]
x_train_set2 = question_set2[:training_samples]
y_train = labels[:training_samples]
len_train_set1 = lengths_set1[:training_samples]
len_train_set2 = lengths_set2[:training_samples]

np.savetxt(train_dir + data1_name, x_train_set1, fmt='%d')
np.savetxt(train_dir + data2_name, x_train_set2, fmt='%d')
np.savetxt(train_dir + labels_name, y_train, fmt='%d')
np.savetxt(train_dir + length1_name, len_train_set1, fmt='%d')
np.savetxt(train_dir + length2_name, len_train_set2, fmt='%d')

x_val_set1 = question_set1[training_samples:]
x_val_set2 = question_set2[training_samples:]
y_val = labels[training_samples:]
len_val = lengths[training_samples:]
len_val_set1 = lengths_set1[training_samples:]
len_val_set2 = lengths_set2[training_samples:]

np.savetxt(val_dir + data1_name, x_val_set1, fmt='%d')
np.savetxt(val_dir + data2_name, x_val_set2, fmt='%d')
np.savetxt(val_dir + labels_name, y_val, fmt='%d')
np.savetxt(val_dir + length1_name, len_val_set1, fmt='%d')
np.savetxt(val_dir + length2_name, len_val_set2, fmt='%d')

# save data, labels, lengths word_index
np.savetxt(data_dir+"data.txt", data, fmt='%d')
np.savetxt(data_dir+"labels.txt", labels, fmt='%d')
np.savetxt(data_dir+"lengths.txt", lengths, fmt='%d')


# In[glove word embeddings]
embeddings_index = {}
with open(os.path.join(glove_dir, "glove.6B.100d.txt"), "r", encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = np.float32)
        embeddings_index[word] = coefs

print('Found %s word vectors.'%len(embeddings_index))

embedding_dim = 100
#embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
embedding_matrix = np.random.rand(len(word_index)+1, embedding_dim)
embedding_matrix = (embedding_matrix-0.5)
embedding_matrix[0] = np.zeros(embedding_dim)

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

np.savetxt(data_dir+"wordvecs.txt", embedding_matrix)






