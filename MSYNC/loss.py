
import tensorflow as tf


def contrastive_loss(y_true, y_pred):
    with tf.device('/device:GPU:0'):
        margin = 1.0
        loss = tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
    return loss


def min_ecl_distance_accuracy(y_true, y_pred):
    with tf.device('/device:GPU:0'):
        min_dist_index_pred = tf.argmin(y_pred, axis=0)
        min_dist_index_true = tf.argmax(y_true, axis=0)
        acc = tf.cast(tf.equal(min_dist_index_true, min_dist_index_pred), tf.float32)
    return acc
