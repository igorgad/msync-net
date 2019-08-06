
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def contrastive_loss(y_true, y_pred):
    y_true = tf.logical_not(y_true)
    with tf.device('/device:CPU:0'):
        margin = 1.0
        loss = tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
    return loss


def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def absolute_range_categorical_accuracy(y_true, y_pred):
    with tf.device('/device:CPU:0'):
        max_dist_index_pred = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=-1), tf.shape(y_pred)[-1]), tf.bool)
        max_dist_index_true = tf.greater(y_true, tf.reduce_min(y_true, axis=-1, keepdims=True))
        range_acc = tf.reduce_any(tf.logical_and(max_dist_index_pred, max_dist_index_true), axis=1)
        result = tf.reduce_mean(tf.cast(range_acc, tf.float32))
    return result


def topn_range_categorical_accuracy(n, range=0):
    def internal_topn_range_categorical_accuracy(y_true, y_pred):
        with tf.device('/device:CPU:0'):
            tops = get_tops(y_pred, n)
            max_dist_index_pred = tf.map_fn(lambda i: tf.cast(tf.reduce_sum(tf.one_hot(i, tf.shape(y_pred)[-1]), axis=0), tf.bool), tops.indices, dtype=tf.bool)

            middle_vals = tf.squeeze(tf.math.top_k(y_true, 1).indices)
            max_dist_index_true = tf.map_fn(lambda val: tf.cast(tf.reduce_sum(tf.one_hot(tf.range(val - range // 2, 1 + val + range // 2), tf.shape(y_pred)[-1]), axis=0), tf.bool), middle_vals, dtype=tf.bool)

            range_acc = tf.reduce_any(tf.logical_and(max_dist_index_pred, max_dist_index_true), axis=1)
            result = tf.reduce_mean(tf.cast(range_acc, tf.float32))
        return result

    func = internal_topn_range_categorical_accuracy
    func.__name__ = 'top%d_range%d_accuracy' % (n, range)
    return func


def zero_descent(signal):
    cut_trail = tf.where(tf.less(tf.gather(signal, tf.range(1, tf.shape(signal)[-1]), axis=-1), tf.gather(signal, tf.range(0, tf.shape(signal)[-1] - 1), axis=-1)),
                         tf.zeros([tf.shape(signal)[0], tf.shape(signal)[-1] - 1], dtype=tf.float32),
                         tf.gather(signal, tf.range(1, tf.shape(signal)[-1]), axis=-1))
    return tf.concat([tf.zeros([tf.shape(cut_trail)[0], 1]), cut_trail], axis=-1)


def get_tops(y_pred, n):
    trail = zero_descent(y_pred)
    trail_rev = tf.reverse(trail, axis=[-1])
    lead = zero_descent(trail_rev)
    lead = tf.reverse(lead, axis=[-1])
    tops = tf.math.top_k(lead, n)
    return tops
