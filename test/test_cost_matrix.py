
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def cost_matrix_func(signals):
    def lin_norm(x, y):
        return tf.norm(x - y, axis=-1)

    os = tf.shape(signals)[-1] // 2
    signal_1 = signals[:, :, 0:os]
    signal_2 = signals[:, :, os:os + os]

    mat = tf.map_fn(lambda ri: lin_norm(tf.expand_dims(signal_1[:, ri, :], axis=1), signal_2[:, :, :]), tf.range(tf.shape(signal_1)[1]), dtype=tf.float32)
    mat = tf.transpose(mat, [1, 0, 2])
    return mat



siga = np.float32(np.random.rand(4, 1024, 256))
sigb = np.float32(np.random.rand(4, 1024, 256))

ksiga = tf.keras.Input(shape=(1024, 256))
ksigb = tf.keras.Input(shape=(1024, 256))
cmb = tf.keras.layers.concatenate([ksiga, ksigb])
cost_mat = tf.keras.layers.Lambda(cost_matrix_func)(cmb)

md = tf.keras.Model([ksiga, ksigb], cost_mat)


r = md.predict([siga, sigb])
