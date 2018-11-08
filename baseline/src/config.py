import os
import tensorflow as tf
from model import BiRNNModel


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
        id_word_name = "id_word_keras.txt"
        wordvecs_name = "wordvecs.txt"

        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")

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

        # id_word path
        self.id_word_path = os.path.join(data_dir, id_word_name)
        
        # wordvecs path
        self.wordvecs_path = os.path.join(data_dir, wordvecs_name)

        '''
        model parameters
        '''
        self.model = BiRNNModel
        self.num_steps = 35
        self.batch_size = 32
        self.wordvec_size = 100
        self.num_words = len(open(self.id_word_path, 'r', encoding='UTF-8').readlines())+1
        # number of hidden nodes in rnn cell
        self.num_units = 200
#        self.atn_hidden_size = self.num_units
        self.mlp_hidden_nodes = [800, 800, 600]

        self.embedding_random_flag = False

        self.initializer = tf.contrib.layers.variance_scaling_initializer
        self.rnn_initializer = tf.glorot_uniform_initializer
        self.optimizer = tf.train.AdamOptimizer

        '''
        training parameters
        '''
        self.lr_decay = True
        self.lr_decay_epoch = 1
        self.lr_decay_rate = 0.9
        self.learning_rate = 0.001
        self.dropout = 0.35

        self.num_epoch = 30
        self.early_stop_epoch = 10
        self.threshold = 1.0
        # model saved flag when meet best valid score
        self.model_save_by_best_valid = True
        # model saved period when model_save_by_best_valid==False
        self.model_save_period = 5
        self.shuffle_data = True

        '''
        model and log paths
        '''
        # model path
        self.model_name = self.model.__name__
        self.model_dir = os.path.join(base_dir, "model")
        self.model_path = os.path.join(self.model_dir, self.model_name)

        # log path
        self.log_dir = os.path.join(base_dir, "logs")
        self.log_train_dir = os.path.join(self.log_dir, "train")
        self.log_train_acc_path = os.path.join(self.log_train_dir, self.model_name+".train.acc.log")
        self.log_valid_acc_path = os.path.join(self.log_train_dir, self.model_name+".valid.acc.log")

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.log_train_dir):
            os.mkdir(self.log_train_dir)