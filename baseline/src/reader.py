
import numpy as np

class Reader:

    def __init__(self, config):
        self.config = config

    def load_data(self):
        data = {}

        # train data
        train_data1 = np.loadtxt(self.config.train_data1_path, dtype = np.int32)
        train_data2 = np.loadtxt(self.config.train_data2_path, dtype = np.int32)
        train_length1 = np.loadtxt(self.config.train_length1_path, dtype = np.int32)
        train_length2 = np.loadtxt(self.config.train_length2_path, dtype = np.int32)
        train_labels = np.loadtxt(self.config.train_labels_path, dtype = np.int32)
        
        train_set = [train_data1, train_data2, train_length1, train_length2, train_labels]

        # valid data
        valid_data1 = np.loadtxt(self.config.valid_data1_path, dtype = np.int32)
        valid_data2 = np.loadtxt(self.config.valid_data2_path, dtype = np.int32)
        valid_length1 = np.loadtxt(self.config.valid_length1_path, dtype = np.int32)
        valid_length2 = np.loadtxt(self.config.valid_length2_path, dtype = np.int32)
        valid_labels = np.loadtxt(self.config.valid_labels_path, dtype = np.int32)
        
        valid_set = [valid_data1, valid_data2, valid_length1, valid_length2, valid_labels]

        data["train"] = train_set
        data["valid"] = valid_set

        return data