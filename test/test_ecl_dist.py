
import tensorflow as tf
import numpy as np
from trainer.Model import EclDistanceMat, DiagMean
import trainer.dataset_interface
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


siga = np.float32(np.random.rand(4, 32, 64))
sigb = siga  #np.float32(np.random.rand(4, 32, 64))

ksiga = tf.keras.Input(shape=(32, 64))
ksigb = tf.keras.Input(shape=(32, 64))

dist_mat = EclDistanceMat()([ksiga, ksigb])
dist_mean = DiagMean()(dist_mat)
dist_max = tf.keras.layers.Softmax()(dist_mean)

md = tf.keras.Model([ksiga, ksigb], [dist_mat, dist_mean, dist_max])
r = md.predict([siga, sigb])

sess = tf.keras.backend.get_session()

y_pred = r[1]
y_true = np.repeat(np.expand_dims(softmax(np.concatenate([np.zeros(14), np.ones(5), np.zeros(14)])), 0), 4, axis=0)

max_dist_index_pred = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=-1), tf.shape(y_pred)[-1]), tf.bool)
max_dist_index_true = tf.greater(y_true, tf.reduce_min(y_true))
range_acc = tf.reduce_any(tf.logical_and(max_dist_index_pred, max_dist_index_true), axis=1)
