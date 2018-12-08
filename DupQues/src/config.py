# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from models import BiRNNModel, DecAtnModel, BiPMModel_FM


class Configuration:

    def __init__(self):

        '''
        data paths
        '''
        data1_name = "data1.txt"
        data2_name = "data2.txt"
        length1_name = "length1.txt"
        length2_name = "length2.txt"
        labels_name = "labels.txt"
        id_word_name = "id_word.txt"
        wordvecs_name = "wordvecs.txt"

        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        data_dir = os.path.join(base_dir, "data")
        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")
        test_dir = os.path.join(data_dir, "test")

        # train path
        self.train_data1_path = os.path.join(train_dir, data1_name)
        self.train_data2_path = os.path.join(train_dir, data2_name)
        self.train_length1_path = os.path.join(train_dir, length1_name)
        self.train_length2_path = os.path.join(train_dir, length2_name)
        self.train_labels_path = os.path.join(train_dir, labels_name)

        # valid path
        self.valid_data1_path = os.path.join(valid_dir, data1_name)
        self.valid_data2_path = os.path.join(valid_dir, data2_name)
        self.valid_length1_path = os.path.join(valid_dir, length1_name)
        self.valid_length2_path = os.path.join(valid_dir, length2_name)
        self.valid_labels_path = os.path.join(valid_dir, labels_name)

        # test path
        self.test_data1_path = os.path.join(test_dir, data1_name)
        self.test_data2_path = os.path.join(test_dir, data2_name)
        self.test_length1_path = os.path.join(test_dir, length1_name)
        self.test_length2_path = os.path.join(test_dir, length2_name)
        self.test_labels_path = os.path.join(test_dir, labels_name)

        # id_word path
        self.id_word_path = os.path.join(data_dir, id_word_name)
        # wordvecs path
        self.wordvecs_path = os.path.join(data_dir, wordvecs_name)
        '''
        model parameters
        '''
        '''
            Siamese-GRU Model
        '''
#        self.model = BiRNNModel
#        self.rnn_units = 200
#        self.mlp_hidden_nodes = [800, 800]
        '''
            Bi-MPM Model
        '''
        self.model = BiPMModel_FM
        self.num_perspectives = 20
#        self.rnn_units = 200
#        self.ag_rnn_units = 200
#        self.mlp_hidden_nodes = [800, 800]
        self.rnn_units = 100
        self.ag_rnn_units = 100
        self.mlp_hidden_nodes = [400, 400]
        '''
            Decomposable Attention Model
        '''
#        self.model = DecAtnModel
#        self.DecAtn_out_dim = 200
#        self.DecAtn_ffn_nodes_F = [100, 100]
#        self.DecAtn_ffn_nodes_G = [200, 200]
#        self.mlp_hidden_nodes = [400, 400]
        '''
            general parameters
        '''
        self.class1_threshold = 0.5
        self.batch_size = 32
        self.num_steps = 35
        self.wordvec_size = 100
        self.num_words = len(open(self.id_word_path, 'r', encoding='UTF-8').readlines())+1
        
        self.word_embedding_pretrained = True
        self.word_embedding_trainable = False
        
        self.initializer = tf.contrib.layers.variance_scaling_initializer
        self.rnn_initializer = tf.glorot_uniform_initializer
#        self.optimizer = tf.train.GradientDescentOptimizer
        self.optimizer = tf.train.AdamOptimizer
        '''
        training parameters
        '''
        self.learning_rate = 2e-4
        self.lr_decay = False
        self.lr_decay_epoch = 5
        self.lr_decay_rate = 0.9
        self.least_lr = 1e-5
        self.dropout = 0.1
        self.emb_keep_prob = 0.99
        
        self.num_epoch = 100
        self.early_stop_epoch = 10
        self.grad_threshold = 1.0
        
        # model saved flag when meet best valid score
        self.model_save_by_best_valid = True
        # model saved period when model_save_by_best_valid==False
        self.model_save_period = 5
        self.shuffle_data = True
        '''
        log paths
        '''
        self.model_name = self.model.__name__
        self.log_dir = os.path.join(base_dir, "logs_"+self.model_name)

        self.model_dir = os.path.join(self.log_dir, "model")
        self.model_path = os.path.join(self.model_dir, self.model_name)

        self.log_graph_dir = os.path.join(self.log_dir, "graph")
        self.log_train_dir = os.path.join(self.log_dir, "train")
        self.log_train_acc_path = os.path.join(self.log_train_dir, self.model_name+".train.acc.log")
        self.log_valid_acc_path = os.path.join(self.log_train_dir, self.model_name+".valid.acc.log")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.log_train_dir):
            os.mkdir(self.log_train_dir)