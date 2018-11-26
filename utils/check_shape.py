# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:36:23 2018

@author: CX
"""
import tensorflow as tf


with tf.Graph().as_default():
    
#    batch_size = 3
#    num_steps = 5
#    rnn_units = 10
#    P_length = tf.constant([3,4,2])
#
##    _idx = tf.tile([tf.range(num_steps)], [batch_size, 1])
##    mask = tf.less(_idx, tf.expand_dims(P_length, 1))
#
#    # (1, num_steps, 1)
#    num_step_expand = tf.expand_dims(tf.expand_dims(tf.range(num_steps), 0), 2)
#    # [batch_size, num_steps, rnn_units]
#    _idx = tf.tile(num_step_expand, [batch_size, 1, rnn_units])
#    length_range_expand = tf.expand_dims(tf.expand_dims(P_length, 1), 2)
#    length_tile = tf.tile(length_range_expand, [1, num_steps, rnn_units])
#    mask = tf.less(_idx, length_tile)
#    
#    
#    PW = tf.constant([[3,4,2],[5,7,8]])
#    Pi_tile2 = tf.constant([[[1,2,3],[4,5,6]], 
#                            [[7,8,9],[10,11,12]], 
#                            [[3,4,2],[5,7,8]]])
#    
#    result = tf.multiply(Pi_tile2, PW)
    
    Q = tf.constant([[[1,2],[3,4],[5,6]], 
                     [[1,2],[3,4],[5,6]], 
                     [[1,2],[3,4],[5,6]]])
    
    similarity = tf.constant([[[1,2,0],[0,0,0]], 
                              [[7,8,9],[10,11,12]], 
                              [[3,4,2],[5,7,8]]])
    
    weight = tf.divide(similarity,
                       tf.reduce_sum(similarity, axis=-1, keepdims=True))

    result = tf.matmul(weight, tf.to_double(Q))

    sess = tf.Session()
#    print(_idx.shape)
#    print(length_tile.shape)
#    print(sess.run(mask))
    print(sess.run(result))
    
