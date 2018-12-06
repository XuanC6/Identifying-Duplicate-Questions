# -*- coding: utf-8 -*-
import tensorflow as tf

'''
BiMPM Layers
'''
def _mask_P(inputs, batch_size, num_steps, out_dim, P_length):
    with tf.variable_scope("mask_P"):
        # [batch_size, num_steps(P)]
        _idx = tf.tile(tf.expand_dims(tf.range(num_steps), 0), [batch_size, 1])
        mask = tf.less(_idx, tf.expand_dims(P_length, 1))
        # [batch_size, num_steps(P), out_dim]
        mask_tile = tf.tile(tf.expand_dims(mask, 2), [1, 1, out_dim])
        
        zero_ph = tf.placeholder(tf.float32, [batch_size, num_steps, out_dim], 
                                 name="zero_ph")
        zeros = tf.zeros_like(zero_ph)
        mask_inputs = tf.where(mask_tile, inputs, zeros)

    return mask_inputs


def Attentive_Matching_Layer(P, Q, P_length, Q_length, shape, initializer=None, 
                             name=None):
    '''
    Attentive_Matching_Layer.
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q: A tensor with shape [batch_size, num_steps, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    '''
    batch_size, num_steps, in_dim, out_dim = shape
    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)

        # [batch_size, num_steps, in_dim]
        P_normalized = tf.nn.l2_normalize(P, axis=-1)
        Q_normalized = tf.nn.l2_normalize(Q, axis=-1)
        '''
        P_length---dim1
        Q_length---dim2
        '''
        # [batch_size, num_steps(P),            1, in_dim]
        # [batch_size,            1, num_steps(Q), in_dim]
        Similarity = tf.multiply(tf.expand_dims(P_normalized, 2),
                                 tf.expand_dims(Q_normalized, 1))
        # [batch_size, num_steps(P), num_steps(Q)]
        similarity = tf.reduce_sum(Similarity, axis=-1)

        # [batch_size, num_steps(P), num_steps(Q)]
        weights = tf.divide(similarity,
                            tf.reduce_sum(similarity, axis=-1, keepdims=True))
                            # [batch_size, num_steps(P), 1]
        mask_weights = _mask_P(weights, batch_size, num_steps, num_steps, P_length)

        # [batch_size, num_steps(P), num_steps(Q)]*[batch_size, num_steps(Q), in_dim]
        # = [batch_size, num_steps(P), in_dim]
        weighted_sum = tf.matmul(mask_weights, Q)

        # [batch_size, num_steps(P),       1, in_dim]
        #                           [out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(weighted_sum, 2), PW), axis=-1)

        # [batch_size, num_steps(P), out_dim]
        result = tf.reduce_sum(tf.multiply(Wv1, Wv2), axis=-1)
#        # deal with P_length
#        mask_matching = _mask_P(result, batch_size, num_steps, out_dim, P_length)
#    return mask_matching
    return result


def Full_Matching_Layer(P, Q_state, P_length, shape, initializer=None, name=None):
    '''
    Full_Matching_Layer
    Args:
        P:       A tensor with shape [batch_size, num_steps, in_dim]
        Q_state: A tensor with shape [batch_size, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    '''
    batch_size, num_steps, in_dim, out_dim = shape
    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)

        # [batch_size, num_steps(P), out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        # [batch_size, out_dim, in_dim]
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(Q_state, 1), PW), axis=-1)

        # [batch_size, num_steps(P), out_dim]
        result = tf.reduce_sum(tf.multiply(Wv1, tf.expand_dims(Wv2, 1)), axis=-1)
#        mask_matching = _mask_P(result, batch_size, num_steps, out_dim, P_length)
#    return mask_matching
    return result


def Maxpooling_Matching_Layer(P, Q, P_length, Q_length, shape, initializer=None, 
                              name=None):
    '''
    Maxpooling_Matching_Layer
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q: A tensor with shape [batch_size, num_steps, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    '''
    batch_size, num_steps, in_dim, out_dim = shape
    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)

        # [batch_size, num_steps(P), out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        # [batch_size, num_steps(Q), out_dim, in_dim]
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(Q, 2), PW), axis=-1)

        # [batch_size, num_steps(P), num_steps(Q), out_dim, (in_dim)]
        _result = tf.reduce_sum(tf.multiply(tf.expand_dims(Wv1, 2), 
                                            tf.expand_dims(Wv2, 1)), axis=-1)
        
        # [batch_size, num_steps(P), out_dim]
        result = tf.reduce_max(_result, axis=2)
    return result


def Max_Attentive_Matching_Layer(P, Q, P_length, Q_length, shape, initializer=None, 
                                 name=None):
    '''
    Max_Attentive_Matching_Layer
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q: A tensor with shape [batch_size, num_steps, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    '''
    batch_size, num_steps, in_dim, out_dim = shape
    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)
        # [batch_size, 1, in_dim]
        max_Q = tf.reduce_max(Q, axis=1, keepdims=True)
        
        # [batch_size, num_steps(P), out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        # [batch_size,            1, out_dim, in_dim]
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(max_Q, 2), PW), axis=-1)

        # [batch_size, num_steps(P), out_dim, (in_dim)]
        result = tf.reduce_sum(tf.multiply(Wv1, Wv2), axis=-1)
    return result


'''
DecAtn Layers
'''
def _mask_dim1(batch_size, dim1, dim2, length):
    # [batch_size, dim1]
    _idx = tf.tile(tf.expand_dims(tf.range(dim1), 0), [batch_size, 1])
    mask = tf.less(_idx, tf.expand_dims(length, 1))
    # [batch_size, dim1, dim2]
    mask_tile = tf.tile(tf.expand_dims(mask, 2), [1, 1, dim2])
    return mask_tile

def _mask_dim2(batch_size, dim1, dim2, length):
    # [batch_size, dim2]
    _idx = tf.tile(tf.expand_dims(tf.range(dim2), 0), [batch_size, 1])
    mask = tf.less(_idx, tf.expand_dims(length, 1))
    # [batch_size, dim1, dim2]
    mask_tile = tf.tile(tf.expand_dims(mask, 1), [1, dim1, 1])
    return mask_tile


def _ffn(P_input, Q_input, shape, nodes_list, dropout, training, activation, initializer):
    batch_size, num_steps, out_dim = shape
    with tf.variable_scope("Feed_Forward"):
        # [batch_size, 2*num_steps, 1, in_dim]
        hidden = tf.expand_dims(tf.concat([P_input, Q_input], axis=1), 2)
        for nodes in nodes_list:
            hidden = tf.layers.conv2d(hidden, nodes, kernel_size=1, strides=(1, 1), 
                                      padding="same", kernel_initializer=initializer)
#            hidden = tf.layers.batch_normalization(hidden, training=training)
            hidden = activation(hidden)
            hidden = tf.layers.dropout(hidden, dropout, 
                                       noise_shape=[batch_size, 1, 1, nodes],
                                       training=training)
        # [batch_size, 2*num_steps, 1, out_dim]
        output = tf.layers.conv2d(hidden, out_dim, kernel_size=1, strides=(1, 1), 
                                  padding="same", kernel_initializer=initializer)
#        output = tf.layers.batch_normalization(output, training=training)
        # [batch_size, 2*num_steps, out_dim]
        output = tf.squeeze(output, [2])
        # [batch_size, num_steps, out_dim]
        P_output, Q_output = tf.split(output, [num_steps, num_steps], 1)

    return P_output, Q_output


def Decomposable_Attention_Layer(P, Q, P_length, Q_length, config, training,
                                 activation=tf.nn.elu, initializer=None, name=None):
    '''
    Decomposable_Attention_Layer
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q: A tensor with shape [batch_size, num_steps, in_dim]
    Return:
        outputs: A tensor with shape [batch_size, num_steps, 2*out_dim]
    '''
    batch_size, num_steps, in_dim, out_dim = config.batch_size, config.num_steps,\
                                             config.wordvec_size, config.DecAtn_out_dim
    with tf.variable_scope(name):
        with tf.variable_scope("Attend"):
            with tf.variable_scope("masks"):
                def create_zeros(shape=[in_dim, out_dim, num_steps]):
                    zeros = []
                    for dim in shape:
                        zero_ph = tf.placeholder(tf.float32, 
                                                 [batch_size, num_steps, dim])
                        zeros.append(tf.zeros_like(zero_ph))
                    return zeros
                zeros_1, zeros_2, zeros_3 = create_zeros()

                mask_dim1_PQ_1 = _mask_dim1(batch_size, num_steps, in_dim, P_length)
                mask_dim1_QP_1 = _mask_dim1(batch_size, num_steps, in_dim, Q_length)
                mask_dim1_PQ_2 = _mask_dim1(batch_size, num_steps, out_dim, P_length)
                mask_dim1_QP_2 = _mask_dim1(batch_size, num_steps, out_dim, Q_length)
                
                mask_dim2_PQ = _mask_dim2(batch_size, num_steps, num_steps, Q_length)
                mask_dim2_QP = _mask_dim2(batch_size, num_steps, num_steps, P_length)

            # [batch_size, num_steps, in_dim]
            F_P, F_Q = _ffn(P, Q, [batch_size, num_steps, in_dim],
                            config.DecAtn_ffn_nodes_F, config.dropout,
                            training, activation, initializer)

            # [batch_size, num_steps(P),            1, in_dim]
            # [batch_size,            1, num_steps(Q), in_dim]
            Logits = tf.multiply(tf.expand_dims(F_P, 2), tf.expand_dims(F_Q, 1))
            # [batch_size, num_steps(P), num_steps(Q)]
            logits_PQ = tf.reduce_sum(Logits, axis=-1)
            # [batch_size, num_steps(Q), num_steps(P)]
            logits_QP = tf.transpose(logits_PQ, perm=[0, 2, 1])
            
            mask_logits_PQ = tf.where(mask_dim2_PQ, logits_PQ, zeros_3)
            mask_logits_QP = tf.where(mask_dim2_QP, logits_QP, zeros_3)
            
            exps_beta = tf.exp(mask_logits_PQ
                               -tf.reduce_max(mask_logits_PQ, axis=2, keepdims=True))
            exps_alpha = tf.exp(mask_logits_QP
                                -tf.reduce_max(mask_logits_QP, axis=2, keepdims=True))
            
            # [batch_size, num_steps(P), num_steps(Q)]
            mask_exps_beta = tf.where(mask_dim2_PQ, exps_beta, zeros_3)
            # [batch_size, num_steps(Q), num_steps(P)]
            mask_exps_alpha = tf.where(mask_dim2_QP, exps_alpha, zeros_3)
            
            # [batch_size, num_steps(P), num_steps(Q)]
            weights_beta = tf.divide(mask_exps_beta,
                                     tf.reduce_sum(mask_exps_beta, axis=2, keepdims=True))
                                     # [batch_size, num_steps(P), 1]
            
            # [batch_size, num_steps(Q), num_steps(P)]
            weights_alpha = tf.divide(mask_exps_alpha,
                                      tf.reduce_sum(mask_exps_alpha, axis=2, keepdims=True))
                                      # [batch_size, num_steps(P), 1]
                                      
            # [batch_size, num_steps, num_steps]
            mask_weights_beta = tf.where(mask_dim2_PQ, weights_beta, zeros_3)
            mask_weights_alpha = tf.where(mask_dim2_QP, weights_alpha, zeros_3)
            
            # [batch_size, num_steps(P), num_steps(Q)]*[batch_size, num_steps(Q), in_dim]
            # = [batch_size, num_steps(P), in_dim]
            Beta = tf.matmul(mask_weights_beta, Q)
            # [batch_size, num_steps(Q), num_steps(P)]*[batch_size, num_steps(P), in_dim]
            # = [batch_size, num_steps(Q), in_dim]
            Alpha = tf.matmul(mask_weights_alpha, P)
            
            mask_Beta = tf.where(mask_dim1_PQ_1, Beta, zeros_1)
            mask_Alpha = tf.where(mask_dim1_QP_1, Alpha, zeros_1)
            
            # [batch_size, num_steps(P), 2*in_dim]
            P_Beta = tf.concat(values=[P, mask_Beta], axis=-1)
            # [batch_size, num_steps(Q), 2*in_dim]
            Q_Alpha = tf.concat(values=[Q, mask_Alpha], axis=-1)

        with tf.variable_scope("Compare"):
            # [batch_size, num_steps, out_dim]
            G_V1, G_V2 = _ffn(P_Beta, Q_Alpha, [batch_size, num_steps, out_dim],
                              config.DecAtn_ffn_nodes_G, config.dropout, 
                              training, activation, initializer)
            # mask G_V1 G_V2
            mask_G_V1 = tf.where(mask_dim1_PQ_2, G_V1, zeros_2)
            mask_G_V2 = tf.where(mask_dim1_QP_2, G_V2, zeros_2)
        
        with tf.variable_scope("Aggregate"):
            # [batch_size, 2*out_dim]
            outputs = tf.concat(values=[tf.reduce_sum(mask_G_V1, axis=1),
                                        tf.reduce_sum(mask_G_V2, axis=1)], axis=-1)
        return outputs