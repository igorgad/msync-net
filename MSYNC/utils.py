
import tensorflow as tf


def range_categorical_accuracy(y_true, y_pred):
    max_dist_index_pred = tf.argmax(y_pred, axis=-1)
    max_dist_index_true = tf.argmax(y_true, axis=-1)
    range_acc = tf.reduce_any(tf.equal(max_dist_index_pred, [max_dist_index_true - 1, max_dist_index_true, max_dist_index_true + 1]), axis=0)
    return tf.cast(range_acc, tf.float32)
