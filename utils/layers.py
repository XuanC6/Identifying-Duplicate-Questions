# -*- coding: utf-8 -*-
import tensorflow as tf


def _mask_result(result, batch_size, num_steps, out_dim, P_length):
    # [batch_size, num_steps(P)]
    _idx = tf.tile(tf.expand_dims(tf.range(num_steps), 0), [batch_size, 1])
    mask = tf.less(_idx, tf.expand_dims(P_length, 1))
    # [batch_size, num_steps(P), out_dim]
    mask_tile = tf.tile(tf.expand_dims(mask, 2), [1, 1, out_dim])
    zeros = tf.zeros([batch_size, num_steps, out_dim])
    mask_matching = tf.where(mask_tile, result, zeros)

    return mask_matching


def Attentive_Matching_Layer(P, Q, P_length, Q_length, shape, initializer=None, 
                             name=None):
    """
    Attentive_Matching_Layer.
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q: A tensor with shape [batch_size, num_steps, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    """
    batch_size, num_steps, in_dim, out_dim = shape

    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)

        # [batch_size, num_steps, in_dim]
        P_normalized = tf.nn.l2_normalize(P, axis=2)
        Q_normalized = tf.nn.l2_normalize(Q, axis=2)
        
        # [batch_size, num_steps(P), num_steps(Q), in_dim]
        Similarity = tf.multiply(tf.expand_dims(P_normalized, 2), 
                                 tf.expand_dims(Q_normalized, 1))
        # [batch_size, num_steps(P), num_steps(Q)]
        similarity = tf.reduce_sum(Similarity, axis=-1)
        '''
        P_length---dim1
        Q_length---dim2
        '''
        # deal with Q_length
        # [batch_size, num_steps(Q)]
        _idx = tf.tile(tf.expand_dims(tf.range(num_steps), 0), [batch_size, 1])
        mask = tf.less(_idx, tf.expand_dims(Q_length, 1))
        # [batch_size, num_steps, num_steps(Q)]
        mask_tile = tf.tile(tf.expand_dims(mask, 1), [1, num_steps, 1])
        zeros = tf.zeros([batch_size, num_steps, num_steps])
        mask_similarity = tf.where(mask_tile, similarity, zeros)

        # [batch_size, num_steps(P), num_steps(Q)]
        weights = tf.div(mask_similarity,
                         tf.reduce_sum(mask_similarity, axis=1, keepdims=True))
                         # [batch_size, 1, num_steps(Q)]

        # [batch_size, num_steps(P), num_steps(Q)]*[batch_size, num_steps(Q), in_dim]
        # = [batch_size, num_steps(P), in_dim]
        weighted_sum = tf.matmul(weights, Q)

        # [batch_size, num_steps(P), 1,       in_dim]
        #                           [out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(weighted_sum, 2), PW), axis=-1)

        # [batch_size, num_steps(P), out_dim]
        result = tf.reduce_sum(tf.multiply(Wv1, Wv2), axis=-1)
        # deal with P_length
        mask_matching = _mask_result(result, batch_size, num_steps, out_dim, P_length)

    return mask_matching


def Full_Matching_Layer(P, Q_state, P_length, shape, initializer=None, name=None):
    """
    Attentive_Matching_Layer.
    Args:
        P: A tensor with shape [batch_size, num_steps, in_dim]
        Q_state: A tensor with shape [batch_size, in_dim]
    Return:
        mask_matching: A tensor with shape [batch_size, num_steps, out_dim]
    """
    batch_size, num_steps, in_dim, out_dim = shape

    with tf.variable_scope(name):
        PW = tf.get_variable(name="Perspective_Weight", shape=[out_dim, in_dim],
                             initializer=initializer)

        # [batch_size, num_steps, out_dim, in_dim]
        Wv1 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(P, 2), PW), axis=-1)
        # [batch_size, out_dim, in_dim]
        Wv2 = tf.nn.l2_normalize(tf.multiply(tf.expand_dims(Q_state, 1), PW), axis=-1)

        # [batch_size, num_steps, out_dim]
        result = tf.reduce_sum(tf.multiply(Wv1, tf.expand_dims(Wv2, 1)), axis=-1)
        mask_matching = _mask_result(result, batch_size, num_steps, out_dim, P_length)

    return mask_matching




#def Attentive_Matching_Layer2(P, Q, P_length, Q_length, shape,
#                              activation=tf.nn.tanh, initializer=None, name=None):
#    """
#    Attentive_Matching_Layer.
#    Args:
#        P: A tensor with shape [batch_size, num_steps, in_dim]
#        Q: A tensor with shape [batch_size, num_steps, in_dim]
#    Return:
#        mask_matching: A tensor with shape [batch_size, num_steps, 2*in_dim]
#    """
#    batch_size, num_steps, in_dim = shape
#
#    with tf.variable_scope(name):
#        AM_W1 = tf.get_variable(name="AM_W1",
#                                shape=[1, 1, 2*in_dim, in_dim],
#                                initializer=initializer)
#        AM_W2 = tf.get_variable(name="AM_W2",
#                                shape=[1, 1, in_dim, 1],
#                                initializer=initializer)
#        AM_b1 = tf.get_variable(name="AM_b1",
#                                shape=[in_dim],
#                                initializer=tf.zeros_initializer())
#        AM_b2 = tf.get_variable(name="AM_b2",
#                                shape=[1],
#                                initializer=tf.zeros_initializer())
#
#        result = [None]*num_steps
#        for i in range(num_steps):
#            # [batch_size, 1, in_dim]
#            Pi = tf.expand_dims(P[:, i, :], 1)
#            # [batch_size, num_steps, in_dim]
#            Pi_tile = tf.tile(Pi, [1, num_steps, 1])
#            # [batch_size, num_steps, 1, 2*in_dim]
#            Pi_Q = tf.expand_dims(tf.concat(values=[Pi_tile, Q], axis=-1), 2)
#            # [batch_size, num_steps, 1, in_dim]
#            activations = activation(tf.nn.conv2d(Pi_Q, AM_W1, 
#                                                  strides=[1, 1, 1, 1],
#                                                  padding="SAME") + AM_b1)
#            # [batch_size, num_steps, 1, 1]
#            scores = tf.nn.conv2d(activations, AM_W2, strides=[1, 1, 1, 1],
#                                  padding="SAME") + AM_b2
#
#            # [batch_size, num_steps]
#            _idx = tf.tile([tf.range(num_steps)], [batch_size, 1])
#            mask = tf.less(_idx, tf.expand_dims(Q_length, 1))
#            zeros = tf.zeros([batch_size, num_steps])
#            mask_logits = tf.where(mask, tf.squeeze(scores, [2, 3]), zeros)
##            exp_logits = tf.exp(mask_logits
##                                - tf.reduce_max(mask_logits, axis=1, keepdims=True))
#            exp_logits = tf.exp(mask_logits)
#            mask_exps = tf.where(mask, exp_logits, zeros)
#            weights = tf.div(mask_exps,
#                             tf.reduce_sum(mask_exps, axis=1, keepdims=True))
#
#            # [batch_size, 1, num_steps]*[batch_size, num_steps, in_dim]
#            # = [batch_size, 1, in_dim]
#            weighted_sum = tf.matmul(tf.expand_dims(weights, 1), Q)
#            # [batch_size, 1, 2*in_dim]
#            result[i] =tf.concat(values=[Pi, weighted_sum], axis=-1) 
#
#        # [batch_size, num_steps, 2*in_dim]
#        matching_result = tf.concat(values=result, axis = 1)
#        # (1, num_steps, 1)
#        num_step_expand = tf.expand_dims(tf.expand_dims(tf.range(num_steps), 0), 2)
#        # [batch_size, num_steps, 2*in_dim]
#        _idx = tf.tile(num_step_expand, [batch_size, 1, 2*in_dim])
#        # [batch_size, 1, 1]
#        length_range_expand = tf.expand_dims(tf.expand_dims(P_length, 1), 2)
#        # [batch_size, num_steps, 2*in_dim]
#        mask = tf.less(_idx, tf.tile(length_range_expand, [1, num_steps, 2*in_dim]))
#        zeros = tf.zeros([batch_size, num_steps, 2*in_dim])
#        mask_matching = tf.where(mask, matching_result, zeros)
#
#    return mask_matching
