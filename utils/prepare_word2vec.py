# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:41:23 2018

@author: CX
"""
import os
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from collections import defaultdict

# In[path]
data_dir = os.path.dirname(os.path.dirname(__file__)) + "/DupQues/data/"
indices_dir = data_dir + "indices/"
train_dir = data_dir + "train/"
val_dir = data_dir + "valid/"
test_dir = data_dir + "test/"

csv_path = data_dir + "questions.csv"
word2vec_model_path = data_dir + "word2vec.model"

if not os.path.exists(indices_dir):
    os.mkdir(indices_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# In[read data]
load_model = False
load_tv = False

# read csv
df = pd.read_csv(csv_path)

old = "′‘’´`"
signs = '?.？。!！ '
split_signs = r'\s*[—…⚡；？！。，、″¨“”：（）《》【】\\!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\s]\s*'

word_index = {}
word_counts = defaultdict(lambda: 0)
documents = []
q_seqs = []
labels = []


def word_processing(q):
    q_seq = []
    q_list_pro = []

    for word in q:
        word = word.strip("' ").lower()
        if word == '' or word == ' ':
            continue
        if word not in word_index:
            word_index[word] = len(word_index)+1
        
        word_counts[word] += 1
        q_seq.append(word_index[word])
        q_list_pro.append(word)

    return q_seq, q_list_pro

# prepare documents for training wv
for index, row in df.iterrows():
    q1 = str(row["question1"])
    q2 = str(row["question2"])
    for s in old:
        q1 = q1.replace(s, "'")
        q2 = q2.replace(s, "'")

    q1_list = re.split(split_signs, q1.strip(signs))
    q2_list = re.split(split_signs, q2.strip(signs))
    
    q1_seq, q1_list_pro = word_processing(q1_list)
    q2_seq, q2_list_pro = word_processing(q2_list)
    
    documents.append(q1_list_pro)
    documents.append(q2_list_pro)

    q_seqs.append(q1_seq)
    q_seqs.append(q2_seq)

    labels.append(int(row["is_duplicate"]))


# In[save dicts and parsed data]
index_word = {value:key for key, value in word_index.items()}
with open(data_dir+"data_parsed.txt","w",encoding='UTF-8') as f:
    for seq in q_seqs:
        for s in seq:
            f.write(index_word[s] + ' ')
        f.write('\n')

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

def pad_seqs(seqs, maxlen):
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

#training_samples = 380000
valid_samples = 10000
test_samples = 10000

# In[save data]
question_set1 = data[::2]
question_set2 = data[1::2]
lengths_set1 = lengths[::2]
lengths_set2 = lengths[1::2]
assert question_set1.shape[0] == question_set2.shape[0]

if os.path.exists(indices_dir+"indices.txt") and load_tv:
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

x_val_set1 = question_set1[:valid_samples]
x_val_set2 = question_set2[:valid_samples]
y_val = labels[:valid_samples]
len_val = lengths[:valid_samples]
len_val_set1 = lengths_set1[:valid_samples]
len_val_set2 = lengths_set2[:valid_samples]

x_test_set1 = question_set1[valid_samples:valid_samples+test_samples]
x_test_set2 = question_set2[valid_samples:valid_samples+test_samples]
y_test = labels[valid_samples:valid_samples+test_samples]
len_test = lengths[valid_samples:valid_samples+test_samples]
len_test_set1 = lengths_set1[valid_samples:valid_samples+test_samples]
len_test_set2 = lengths_set2[valid_samples:valid_samples+test_samples]

x_train_set1 = question_set1[valid_samples+test_samples:]
x_train_set2 = question_set2[valid_samples+test_samples:]
y_train = labels[valid_samples+test_samples:]
len_train_set1 = lengths_set1[valid_samples+test_samples:]
len_train_set2 = lengths_set2[valid_samples+test_samples:]

np.savetxt(train_dir + data1_name, x_train_set1, fmt='%d')
np.savetxt(train_dir + data2_name, x_train_set2, fmt='%d')
np.savetxt(train_dir + labels_name, y_train, fmt='%d')
np.savetxt(train_dir + length1_name, len_train_set1, fmt='%d')
np.savetxt(train_dir + length2_name, len_train_set2, fmt='%d')

np.savetxt(val_dir + data1_name, x_val_set1, fmt='%d')
np.savetxt(val_dir + data2_name, x_val_set2, fmt='%d')
np.savetxt(val_dir + labels_name, y_val, fmt='%d')
np.savetxt(val_dir + length1_name, len_val_set1, fmt='%d')
np.savetxt(val_dir + length2_name, len_val_set2, fmt='%d')

np.savetxt(test_dir + data1_name, x_test_set1, fmt='%d')
np.savetxt(test_dir + data2_name, x_test_set2, fmt='%d')
np.savetxt(test_dir + labels_name, y_test, fmt='%d')
np.savetxt(test_dir + length1_name, len_test_set1, fmt='%d')
np.savetxt(test_dir + length2_name, len_test_set2, fmt='%d')

# save data, labels, lengths word_index
np.savetxt(data_dir+"data.txt", data, fmt='%d')
np.savetxt(data_dir+"labels.txt", labels, fmt='%d')
np.savetxt(data_dir+"lengths.txt", lengths, fmt='%d')

# In[train wv on train set]
documents_set1 = documents[::2]
documents_set2 = documents[1::2]
documents_set1 = [documents_set1[idx] for idx in indices]
documents_set2 = [documents_set2[idx] for idx in indices]
documents_train_set1 = documents_set1[valid_samples+test_samples:]
documents_train_set2 = documents_set2[valid_samples+test_samples:]
documents_train = documents_train_set1+documents_train_set2

if load_model:
    model = Word2Vec.load(word2vec_model_path)
else:
    model = Word2Vec(size=100, window=10, min_count=2, sg=1, workers=4)
    model.iter = 10
    model.build_vocab(documents_train)  # prepare the model vocabulary
    model.train(sentences=documents_train, total_examples=len(documents_train),
                epochs=model.iter)
    model.save(word2vec_model_path)

# In[get word embeddings]
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
#embedding_matrix = np.random.rand(len(word_index)+1, embedding_dim)
#embedding_matrix = (embedding_matrix-0.5)
#embedding_matrix[0] = np.zeros(embedding_dim)

for word, i in word_index.items():
    try:
        embedding_vector = model.wv.get_vector(word)
    except KeyError:
        embedding_vector = None

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

np.savetxt(data_dir+"wordvecs.txt", embedding_matrix)

