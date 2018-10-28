
import tensorflow as tf


class BiGRUModel:

    def __init__(self, config):
        self.config = config

        self.num_steps = config.num_steps
        self.num_words = config.num_words
        self.wordvec_size = config.wordvec_size
        self.batch_size = config.batch_size
        self.initializer = config.initializer

        self.add_placeholders()
        inputs = self.add_embedding()
        self.compute_loss(inputs)
        self.add_train_op()


    def add_placeholders(self):
        self.input1 = tf.placeholder(tf.int32, [None, self.num_steps], name="ques1")
        self.input2 = tf.placeholder(tf.int32, [None, self.num_steps], name="ques2")
        self.length1 = tf.placeholder(tf.int32, [None], name="length1")
        self.length2 = tf.placeholder(tf.int32, [None], name="length2")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")


    def add_embedding(self):
        with tf.variable_scope("Embed"):
            if self.config.embedding_random_flag:
                embedding1 = tf.get_variable(
                    'embedding1',
                    [self.num_words, self.wordvec_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05),
                    trainable = True,
                )
                embedding2 = tf.get_variable(
                    'embedding2',
                    [self.num_words, self.wordvec_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05),
                    trainable = True,
                )
            else:
                embedding1 = tf.get_variable("embedding1", 
                                             [self.num_words, self.wordvec_size],
                                             trainable = True)
                embedding2 = tf.get_variable("embedding2", 
                                             [self.num_words, self.wordvec_size],
                                             trainable = True)

            vec_input1 = tf.nn.embedding_lookup(embedding1, self.input1)
            vec_input2 = tf.nn.embedding_lookup(embedding2, self.input2)

        return [vec_input1, vec_input2]


    def _run_rnn(self, inputs):
        #calculate outputs
        with tf.variable_scope("GRU1"):
            cell_fw_1 = tf.nn.rnn_cell.GRUCell(num_units=self.config.n_neurons)
            cell_bw_1 = tf.nn.rnn_cell.GRUCell(num_units=self.config.n_neurons)
        with tf.variable_scope("GRU2"):
            cell_fw_2 = tf.nn.rnn_cell.GRUCell(num_units=self.config.n_neurons)
            cell_bw_2 = tf.nn.rnn_cell.GRUCell(num_units=self.config.n_neurons)

        with tf.variable_scope("GRU1_bi"):
            states1, output1 = tf.nn.bidirectional_dynamic_rnn(cell_fw_1, cell_bw_1, inputs[0],
                                                               sequence_length = self.length1,
                                                               dtype=tf.float32)
        with tf.variable_scope("GRU2_bi"):
            states2, output2 = tf.nn.bidirectional_dynamic_rnn(cell_fw_2, cell_bw_2, inputs[1],
                                                               sequence_length = self.length2,
                                                               dtype=tf.float32)

        fw_out1, bw_out1 = output1
        fw_out2, bw_out2 = output2
        
        rnn_output = tf.concat([fw_out1, bw_out1, fw_out2, bw_out2], axis = -1)
        
        return rnn_output



    def compute_loss(self, inputs):
        rnn_output = self._run_rnn(inputs)
        
        n_hidden_nodes = [600, 600, 400]
        hidden_names = ['hidden1','hidden2','hidden3']
        hidden_layer = rnn_output
        
        for n_node, name in zip(n_hidden_nodes, hidden_names):
            hidden_layer = tf.layers.dense(hidden_layer, n_node, 
                                           activation = tf.nn.elu,
                                           kernel_initializer = self.initializer,
                                           name = name)

        logits = tf.layers.dense(hidden_layer, 1,
                                 kernel_initializer = self.initializer,
                                 name = 'logits')

        scores = tf.nn.sigmoid(logits, name="predict_probs")
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.to_float(self.labels),
                                                       logits = tf.squeeze(logits))
        self.loss = tf.reduce_mean(loss, name="loss")

        result_options = tf.concat([1-scores, scores], axis = 1)
        self.predicts = tf.argmax(result_options, axis = 1)
        self.probabilities = tf.reduce_max(result_options, axis = 1)


    def add_train_op(self):
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
        optimizer = self.config.optimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.config.threshold, self.config.threshold), var)
                        for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, name="train")







