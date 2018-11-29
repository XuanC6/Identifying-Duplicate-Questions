# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

from layers import Attentive_Matching_Layer, Full_Matching_Layer, \
                    Maxpooling_Matching_Layer, Max_Attentive_Matching_Layer, \
                    Decomposable_Attention_Layer



class BiRNNModel:
    '''
    Siamese_GRU
    '''
    def __init__(self, config):
        self.config = config

        self.num_steps = config.num_steps
        self.num_words = config.num_words
        self.wordvec_size = config.wordvec_size
        self.batch_size = config.batch_size
        self.initializer = config.initializer
        self.rnn_initializer = config.rnn_initializer

        self.add_placeholders()
        inputs = self.add_embedding()
        self.compute_loss(inputs)
        self.add_train_op()


    def add_placeholders(self):
        dim1 = self.batch_size
        dim2 = self.num_steps
        
        self.input1 = tf.placeholder(tf.int32, [dim1, dim2], name="ques1")
        self.input2 = tf.placeholder(tf.int32, [dim1, dim2], name="ques2")
        self.length1 = tf.placeholder(tf.int32, [dim1], name="length1")
        self.length2 = tf.placeholder(tf.int32, [dim1], name="length2")
        self.labels = tf.placeholder(tf.int32, [dim1], name="labels")
        self.keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")
        self.training = tf.placeholder_with_default(False, [], name="training")


    def add_embedding(self):
        def add_embedding_helper(name):
            embedding = tf.get_variable(
                name,
                [self.num_words, self.wordvec_size],
                initializer = tf.random_uniform_initializer(-0.5, 0.5),
                trainable = self.config.word_embedding_trainable)
            return embedding
        
        with tf.variable_scope("Embed"):
            embedding1 = add_embedding_helper("embedding1")
            embedding2 = add_embedding_helper("embedding2")
        
        vec_input1 = tf.nn.embedding_lookup(embedding1, self.input1)
        vec_input2 = tf.nn.embedding_lookup(embedding2, self.input2)

        return [vec_input1, vec_input2]


    def _create_rnncells(self, n, num_units):
        rnncells = [None for _ in range(n)]
        for i in range(n):
             _rnncell = tf.nn.rnn_cell.GRUCell(num_units=num_units,
                                               kernel_initializer=self.rnn_initializer())
             rnncells[i] = tf.nn.rnn_cell.DropoutWrapper(_rnncell,
                                                         output_keep_prob=self.keep_prob)
#            rnncells[i] = tf.nn.rnn_cell.GRUCell(num_units=num_units,
#                                                 kernel_initializer=self.rnn_initializer())
        return rnncells


    def _run_core(self, inputs):
        # calculate output
        rnncells = self._create_rnncells(4, self.config.rnn_units)
        with tf.variable_scope("Bidirectional_RNN1"):
            _, state1 = tf.nn.bidirectional_dynamic_rnn(rnncells[0], rnncells[1], inputs[0],
                                                        sequence_length=self.length1,
                                                        dtype=tf.float32)
        with tf.variable_scope("Bidirectional_RNN2"):
            _, state2 = tf.nn.bidirectional_dynamic_rnn(rnncells[2], rnncells[3], inputs[1],
                                                        sequence_length=self.length2,
                                                        dtype=tf.float32)

        fw1, bw1 = state1
        fw2, bw2 = state2
        core_output = tf.concat(values = [fw1, bw1, fw2, bw2], axis = -1)
        
        return core_output


    def compute_loss(self, inputs):
        hidden, mask_Beta = self._run_core(inputs)
        self.hidden = mask_Beta
        
        with tf.variable_scope("Dense_Layers"):
            for n_node in self.config.mlp_hidden_nodes:
                hidden = tf.layers.dense(hidden, n_node, 
                                         kernel_initializer=self.initializer())
                hidden = tf.nn.elu(hidden)
                hidden = tf.layers.dropout(hidden, self.config.dropout,
                                           training=self.training)

        with tf.variable_scope("Loss"):
            logits = tf.layers.dense(hidden, 1,
                                     kernel_initializer = self.initializer(),
                                     name = 'logits')

            self.scores = tf.nn.sigmoid(logits, name="predict_probs")
            float_labels=tf.to_float(self.labels)
            clipped_scores = tf.clip_by_value(tf.squeeze(self.scores), 1e-6, 1-1e-6)
            loss = -tf.multiply(float_labels, tf.log(clipped_scores))-\
                    tf.multiply((1.0-float_labels), tf.log(1.0-clipped_scores))

#            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.labels),
#                                                           logits=tf.squeeze(logits))

#            self.scores = tf.nn.softmax(logits, name="predict_probs")
#            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.labels, 2),
#                                                              logits=logits)

#            # put more weight on samples with label 1
#            alpha = 0.266
#            weights = tf.exp((tf.to_float(self.labels)-0.5)*2*alpha, name="weights")
#            weighted_loss = tf.multiply(weights, loss, name="weighted_loss")
#            self.loss = tf.reduce_sum(weighted_loss)/tf.reduce_sum(weights)

            # normal loss
            self.loss = tf.reduce_mean(loss)
            # sigmoid
            result_options = tf.concat([1-self.scores, self.scores], axis=1)
            self.predicts = tf.argmax(result_options, axis = 1)
            self.probabilities = tf.reduce_max(result_options, axis=1)
#            # softmax
#            self.predicts = tf.argmax(self.scores, axis = 1)
#            self.probabilities = tf.reduce_max(self.scores, axis=1)


    def add_train_op(self):
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
        optimizer = self.config.optimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        with tf.variable_scope("clip_grad"):
            capped_gvs = [(tf.clip_by_value(grad, -self.config.grad_threshold,
                                            self.config.grad_threshold), var)
                            for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, name="train_op")


class BiPMModel2(BiRNNModel):
    '''
    Attentive_Matching
    '''
    def _context_representation(self, rnncells, inputs):
        with tf.variable_scope("Context_Representation_Layer1"):
            outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(rnncells[0], rnncells[1], inputs[0],
                                                                sequence_length=self.length1,
#                                                               swap_memory=True,
                                                                dtype=tf.float32)
        with tf.variable_scope("Context_Representation_Layer2"):
            outputs2, states2 = tf.nn.bidirectional_dynamic_rnn(rnncells[2], rnncells[3], inputs[1],
                                                                sequence_length=self.length2,
#                                                               swap_memory=True,
                                                                dtype=tf.float32)
        return outputs1, outputs2, states1, states2


    def _create_matching_layer(self, P_input, Q_input, P_length, Q_length, name=None):
        _matching = Attentive_Matching_Layer(P_input, Q_input, P_length, Q_length,
                                             shape=[self.batch_size, self.num_steps, 
                                                    self.config.rnn_units, 
                                                    self.config.num_perspectives],
                                             initializer=self.initializer(),
                                             name="AM_"+name)
        matching = tf.layers.dropout(_matching, self.config.dropout,
                                     training=self.training)
        return matching


    def _matching(self, **inputs):
        P_outputs = inputs["P_outputs"]
        Q_outputs = inputs["Q_outputs"]
        
        # [batch_size, num_steps, rnn_units]
        P_fw, P_bw = P_outputs
        Q_fw, Q_bw = Q_outputs

        with tf.variable_scope("Matching_Layer_PQ"):
            # [batch_size, num_steps, num_perspectives]
            matching_PQ_fw = self._create_matching_layer(P_fw, Q_fw,
                                                   self.length1, self.length2, "PQ_fw")
            matching_PQ_bw = self._create_matching_layer(P_bw, Q_bw, 
                                                   self.length1, self.length2, "PQ_bw")

        with tf.variable_scope("Matching_Layer_QP"):
            # [batch_size, num_steps, num_perspectives]
            matching_QP_fw = self._create_matching_layer(Q_fw, P_fw,
                                                   self.length2, self.length1, "QP_fw")
            matching_QP_bw = self._create_matching_layer(Q_bw, P_bw,
                                                   self.length2, self.length1, "QP_bw")

        # [batch_size, num_steps, 2*num_perspectives]
        matching_PQ = tf.concat(values=[matching_PQ_fw, matching_PQ_bw], axis=-1)
        matching_QP = tf.concat(values=[matching_QP_fw, matching_QP_bw], axis=-1)
        
        return matching_PQ, matching_QP


    def _aggregation(self, rnncells, matching_PQ, matching_QP):
        with tf.variable_scope("Aggregation_Layer1"):
            _, state1 = tf.nn.bidirectional_dynamic_rnn(rnncells[0], rnncells[1], matching_PQ,
                                                        sequence_length=self.length1,
#                                                        swap_memory=True,
                                                        dtype=tf.float32)
        with tf.variable_scope("Aggregation_Layer2"):
            _, state2 = tf.nn.bidirectional_dynamic_rnn(rnncells[2], rnncells[3], matching_QP,
                                                        sequence_length=self.length2,
#                                                        swap_memory=True,
                                                        dtype=tf.float32)
        # [batch_size, num_steps, ag_rnn_units]
        fw1, bw1 = state1
        fw2, bw2 = state2
        # [batch_size, 4*ag_rnn_units]
        return tf.concat(values=[fw1, bw1, fw2, bw2], axis=-1)


    def _run_core(self, inputs):
        # calculate outputs
        cr_rnncells = self._create_rnncells(4, self.config.rnn_units)
        ag_rnncells = self._create_rnncells(4, self.config.ag_rnn_units)

        outputs1, outputs2, states1, states2 = self._context_representation(cr_rnncells, inputs)
        matching_PQ, matching_QP = self._matching(P_outputs=outputs1, Q_outputs=outputs2, 
                                                  P_states=states1, Q_states=states2)
        core_output = self._aggregation(ag_rnncells, matching_PQ, matching_QP)
#        print(outputs1[0].shape)
#        print(mask_matching_PQ.shape)
#        print(core_output.shape)
        return core_output



class BiPMModel4(BiPMModel2):
    '''
    Maxpooling_Matching
    '''
    def _create_matching_layer(self, P_input, Q_input, P_length, Q_length, name=None):
        _matching = Maxpooling_Matching_Layer(P_input, Q_input, P_length, Q_length,
                                             shape=[self.batch_size, self.num_steps, 
                                                    self.config.rnn_units, 
                                                    self.config.num_perspectives],
                                             initializer=self.initializer(),
                                             name="MM_"+name)
        matching = tf.layers.dropout(_matching, self.config.dropout,
                                     training=self.training)
        return matching



class BiPMModel5(BiPMModel2):
    '''
    Max_Attentive_Matching
    '''
    def _create_matching_layer(self, P_input, Q_input, P_length, Q_length, name=None):
        _matching =  Max_Attentive_Matching_Layer(P_input, Q_input, P_length, Q_length,
                                                  shape=[self.batch_size, self.num_steps, 
                                                         self.config.rnn_units, 
                                                         self.config.num_perspectives],
                                                  initializer=self.initializer(),
                                                  name="MAM_"+name)
        matching = tf.layers.dropout(_matching, self.config.dropout,
                                     training=self.training)
        return matching



class BiPMModel3(BiPMModel2):
    '''
    Full_Matching
    '''
    def _create_matching_layer(self, P_input, Q_input, P_length, name=None):
        _matching = Full_Matching_Layer(P_input, Q_input, P_length,
                                       shape=[self.batch_size, self.num_steps, 
                                              self.config.rnn_units, 
                                              self.config.num_perspectives],
                                        initializer=self.initializer(),
                                        name="FM_"+name)
        matching = tf.layers.dropout(_matching, self.config.dropout,
                                     training=self.training)
        return matching
    
    def _matching(self, **inputs):
        P_outputs = inputs["P_outputs"]
        Q_outputs = inputs["Q_outputs"]
        P_states = inputs["P_states"]
        Q_states = inputs["Q_states"]
        
        # [batch_size, num_steps, rnn_units]
        P_fw, P_bw = P_outputs
        Q_fw, Q_bw = Q_outputs
        # [batch_size, rnn_units]
        P_state_fw, P_state_bw = P_states
        Q_state_fw, Q_state_bw = Q_states

        with tf.variable_scope("Matching_Layer_PQ"):
            # [batch_size, num_steps, num_perspectives]
            matching_PQ_fw = self._create_matching_layer(P_fw, Q_state_fw, 
                                                   self.length1, "PQ_fw")
            matching_PQ_bw = self._create_matching_layer(P_bw, Q_state_bw, 
                                                   self.length1, "PQ_bw")

        with tf.variable_scope("Matching_Layer_QP"):
            # [batch_size, num_steps, num_perspectives]
            matching_QP_fw = self._create_matching_layer(Q_fw, P_state_fw, 
                                                   self.length2, "QP_fw")
            matching_QP_bw = self._create_matching_layer(Q_bw, P_state_bw, 
                                                   self.length2, "QP_bw")

        # [batch_size, num_steps, 2*num_perspectives]
        matching_PQ = tf.concat(values=[matching_PQ_fw, matching_PQ_bw], axis=-1)
        matching_QP = tf.concat(values=[matching_QP_fw, matching_QP_bw], axis=-1)
        
        return matching_PQ, matching_QP



class DecAtnModel(BiRNNModel):
    '''
    Decomposable Attention
    '''
    def _run_core(self, inputs):
        # calculate output
        core_output, mask_Beta = Decomposable_Attention_Layer(inputs[0], inputs[1],
                                                   self.length1, self.length2,
                                                   config=self.config,
                                                   training=self.training,
                                                   initializer=self.initializer(),
                                                   name="DecAtn_Layer")
        # [batch_size, 2*out_dim]
#        print(core_output.shape)
        return core_output, mask_Beta


#    def compute_loss(self, inputs):
##        hidden = self._run_core(inputs)
#        hidden, mask_Beta = self._run_core(inputs)
#        self.hidden = mask_Beta
#        
#        with tf.variable_scope("Dense_Layers"):
#            for n_node in self.config.mlp_hidden_nodes:
#                hidden = tf.layers.dense(hidden, n_node, 
#                                         kernel_initializer=self.initializer())
#                hidden = tf.layers.batch_normalization(hidden, training=self.training)
#                hidden = tf.nn.elu(hidden)
#                hidden = tf.layers.dropout(hidden, self.config.dropout, 
#                                           training=self.training)
#
#        with tf.variable_scope("Loss"):
#            logits = tf.layers.dense(hidden, 1, kernel_initializer = self.initializer())
#            logits = tf.layers.batch_normalization(logits, training=self.training,
#                                                   name = 'logits')
#
#            self.scores = tf.nn.sigmoid(logits, name="predict_probs")
#            float_labels=tf.to_float(self.labels)
#            clipped_scores = tf.clip_by_value(tf.squeeze(self.scores), 1e-6, 1-1e-6)
#            loss = -tf.multiply(float_labels, tf.log(clipped_scores))-\
#                    tf.multiply((1.0-float_labels), tf.log(1.0-clipped_scores))
#
##            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.labels),
##                                                           logits=tf.squeeze(logits))
#
#            # normal loss
#            self.loss = tf.reduce_mean(loss)
#            # sigmoid
#            result_options = tf.concat([1-self.scores, self.scores], axis=1)
#            self.predicts = tf.argmax(result_options, axis = 1)
#            self.probabilities = tf.reduce_max(result_options, axis=1)
#
#
#    def add_train_op(self):
#        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
#        optimizer = self.config.optimizer(self.learning_rate)
#        
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops):
#            self.train_op = optimizer.minimize(self.loss, name="train_op")


