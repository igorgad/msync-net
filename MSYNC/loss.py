
import tensorflow as tf


def contrastive_loss(y_true, y_pred):
    margin = 2.0
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.reduce_max(margin - y_pred, 0)))
