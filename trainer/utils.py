
import tensorflow as tf


def contrastive_loss(y_true, y_pred):
    y_true = tf.logical_not(y_true)
    with tf.device('/device:GPU:0'):
        margin = 1.0
        loss = tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
    return loss


def range_categorical_accuracy(y_true, y_pred):
    max_dist_index_pred = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=-1), tf.shape(y_pred)[-1]), tf.bool)
    max_dist_index_true = tf.greater(y_true, 0.0)
    range_acc = tf.reduce_any(tf.logical_and(max_dist_index_pred, max_dist_index_true), axis=1)
    return tf.cast(range_acc, tf.float32)
