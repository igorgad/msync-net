
import tensorflow as tf
import numpy as np
from trainer.Model import DiagMean
# import dataset_interface
import matplotlib.pyplot as plt


def diag_mean(mat, diagi):
    ny = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1] - 1, tf.abs(tf.maximum(diagi, 0))))
    nx = tf.add(ny, diagi)
    flat_indices = ny * tf.shape(mat)[1] + nx
    flat_mat = tf.reshape(mat, [tf.shape(mat)[0], -1])
    mean = tf.reduce_mean(tf.gather(flat_mat, flat_indices, axis=1), axis=1)
    inv_mean = tf.reduce_max(mean, axis=-1) - mean
    return inv_mean


inputs = np.float32(np.random.rand(4, 32, 32))

num_time_steps = tf.shape(inputs)[1]
mean = tf.map_fn(lambda ts: diag_mean(inputs, ts), tf.range(-num_time_steps//2, 1 + num_time_steps//2), dtype=tf.float32)
mean = tf.transpose(mean)

sess = tf.keras.backend.get_session()
r = sess.run(mean)

