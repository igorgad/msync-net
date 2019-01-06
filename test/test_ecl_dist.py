
import tensorflow as tf
import numpy as np
from trainer.Model import EclDistanceMat, DiagMean
import trainer.dataset_interface
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0, tf.pow(tf.subtract(x,y), 2)), tf.multiply(2.0, tf.pow(s, 2))) )


def zero_descent(signal, index):
    return tf.where(tf.gather(signal, index - 1, axis=-1) < tf.gather(signal, index, axis=-1), tf.gather(signal, index, axis=-1), tf.zeros(tf.shape(signal)[0], dtype=tf.float32))


def skeletonize_1d(tens):
    initializer = (np.array(0, dtype=np.float32), np.array(0, dtype=np.float32))
    trail = tf.scan(zero_descent, tens, initializer)
    trail_rev = tf.reverse(trail[1], [0])
    lead = tf.scan(zero_descent, trail_rev, initializer)
    return tf.reverse(lead[1], [0])

def find_local_maxima(tens, nmax):
    peak_idx = tf.squeeze(tf.where(skeletonize_1d(tens) > 0))
    peak_val = tf.gather(tens, peak_idx)
    tops = tf.math.top_k(peak_val, nmax)
    return tf.gather(peak_idx, tops.indices)



siga = np.float32(np.random.rand(4, 32, 64))
sigb = siga  #np.float32(np.random.rand(4, 32, 64))

ksiga = tf.keras.Input(shape=(32, 64))
ksigb = tf.keras.Input(shape=(32, 64))

dist_mat = EclDistanceMat()([ksiga, ksigb])
dist_mean = DiagMean()(dist_mat)
dist_max = tf.keras.layers.Softmax()(dist_mean)

md = tf.keras.Model([ksiga, ksigb], [dist_mat, dist_mean, dist_max])
r = md.predict([siga, sigb])
y_pred = r[1]

sess = tf.keras.backend.get_session()

n = 3
x = np.arange(-16, 17, dtype=np.float32)
g1 = sess.run(gkernel(x, 0.0, 2.0))
g2 = sess.run(gkernel(x, 8.0, 2.0)) + g1
g3 = sess.run(gkernel(x, -8.0, 2.0)) + g2
g4 = sess.run(gkernel(x, 13.0, 1.0)) + g3


y_true = np.concatenate([np.expand_dims(g1, 0), np.expand_dims(g2, 0), np.expand_dims(g3, 0), np.expand_dims(g4, 0)], axis=0)
# y_true = np.repeat(np.expand_dims(softmax(np.concatenate([np.zeros(14), np.ones(5), np.zeros(14)])), 0), 4, axis=0)


trail = tf.transpose(tf.map_fn(lambda i: zero_descent(y_true, i), tf.range(1, tf.shape(y_true)[-1]), dtype=tf.float32), [1, 0])
trail_rev = tf.reverse(trail, axis=[-1])
lead = tf.transpose(tf.map_fn(lambda i: zero_descent(trail_rev, i), tf.range(1, tf.shape(trail_rev)[-1]), dtype=tf.float32), [1, 0])
lead = tf.reverse(lead, axis=[-1])
tops = tf.math.top_k(lead, n)
max_dist_index_pred = tf.map_fn(lambda i: tf.cast(tf.reduce_sum(tf.one_hot(i, tf.shape(y_pred)[-1]), axis=0), tf.bool), tops.indices, dtype=tf.bool)
max_dist_index_true = tf.greater(y_true, tf.reduce_min(y_true))
range_acc = tf.reduce_any(tf.logical_and(max_dist_index_pred, max_dist_index_true), axis=1)

