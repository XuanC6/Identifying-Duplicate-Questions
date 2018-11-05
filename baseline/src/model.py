
import tensorflow as tf


class BiLSTMModel:

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
        self.training = tf.placeholder_with_default(False, [], name="training")


    def add_embedding(self):
        def add_embedding_helper(name):
            embedding = tf.get_variable(
                name,
                [self.num_words, self.wordvec_size],
                initializer = tf.random_uniform_initializer(-0.05, 0.05),
                trainable = True)
            return embedding
        
        with tf.variable_scope("Embed"):
            embedding1 = add_embedding_helper("embedding1")
            embedding2 = add_embedding_helper("embedding2")
        
        vec_input1 = tf.nn.embedding_lookup(embedding1, self.input1)
        vec_input2 = tf.nn.embedding_lookup(embedding2, self.input2)

        return [vec_input1, vec_input2]


    def _run_rnn(self, inputs):
        #calculate outputs
        def get_lstmcells():
            lstmcells = [None for _ in range(4)]
            for i in range(4):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.num_units,
                                               use_peepholes=True,
                                               state_is_tuple=True)
                cell = tf.contrib.rnn.AttentionCellWrapper(cell,
                                                           attn_length=self.num_steps)
                lstmcells[i] = cell
            return lstmcells

        lstmcells = get_lstmcells()
        with tf.variable_scope("Bidirectional_LSTM1"):
            outputs1, state1 = tf.nn.bidirectional_dynamic_rnn(lstmcells[0], lstmcells[1], inputs[0],
                                                               sequence_length = self.length1,
                                                               dtype=tf.float32)
        with tf.variable_scope("Bidirectional_LSTM2"):
            outputs2, state2 = tf.nn.bidirectional_dynamic_rnn(lstmcells[2], lstmcells[3], inputs[1],
                                                               sequence_length = self.length2,
                                                               dtype=tf.float32)

        fw_state1, bw_state1 = state1
        fw_state2, bw_state2 = state2
        
        rnn_output = tf.concat([fw_state1[1], bw_state1[1], 
                                fw_state2[1], bw_state2[1]],
                                axis = -1)
        
        return rnn_output


    def compute_loss(self, inputs):
        hidden_layer = self._run_rnn(inputs)
        
        for n_node in self.config.mlp_hidden_nodes:
            hidden_layer = tf.layers.dense(hidden_layer, n_node, 
                                           activation = tf.nn.elu,
                                           kernel_initializer = self.initializer)
            hidden_layer = tf.layers.dropout(hidden_layer, self.config.dropout,
                                             training = self.training)

        logits = tf.layers.dense(hidden_layer, 1,
                                 kernel_initializer = self.initializer,
                                 name = 'logits')

        scores = tf.nn.sigmoid(logits, name="predict_probs")
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.to_float(self.labels),
                                                       logits = tf.squeeze(logits))
        
        # put more weight on samples with label 1
        alpha = 0.266
        weights = tf.exp((tf.to_float(self.labels)-0.5)*2*alpha, name="weights")
        weighted_loss = tf.multiply(weights, loss, name="weighted_loss")
        self.loss = tf.reduce_sum(weighted_loss)/tf.reduce_sum(weights)

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







