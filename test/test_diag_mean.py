
import tensorflow as tf
import numpy as np
from MSYNC.Model import EclDistanceMat
import dataset_interface
import matplotlib.pyplot as plt


def diag_mean(mat, diagi):
    nx = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1], tf.abs(tf.maximum(diagi, 0))))
    ny = tf.add(nx, diagi)
    indices = [tf.concat([tf.expand_dims(nx, -1), tf.expand_dims(ny, -1)], axis=1)]
    return tf.reduce_mean(tf.gather_nd(mat, indices))


mat = np.float32(np.random.rand(4, 32, 32, 1))
inputs = mat
num_time_steps = tf.shape(inputs)[1]
# mean = tf.map_fn(lambda ts: diag_mean(inputs, ts), tf.range(-num_time_steps + 1, num_time_steps), dtype=tf.float32)

diagi = 0

nx = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1], tf.abs(tf.maximum(diagi, 0))))
ny = tf.add(nx, diagi)
indices = [tf.concat([tf.expand_dims(nx, -1), tf.expand_dims(ny, -1)], axis=1)]
vec = tf.gather_nd(mat, indices)

sess = tf.Session()
i, r = sess.run([indices, vec])