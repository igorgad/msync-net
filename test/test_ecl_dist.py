
import tensorflow as tf
import numpy as np
from MSYNC.Model import EclDistanceMat, DiagMean
import dataset_interface
import matplotlib.pyplot as plt


siga = np.float32(np.random.rand(4, 32, 64))
sigb = siga  #np.float32(np.random.rand(4, 32, 64))

ksiga = tf.keras.Input(shape=(32, 64))
ksigb = tf.keras.Input(shape=(32, 64))

dist_mat = EclDistanceMat()([ksiga, ksigb])
dist_mean = DiagMean()(dist_mat)
dist_max = tf.keras.layers.Softmax()(dist_mean)

md = tf.keras.Model([ksiga, ksigb], [dist_mat, dist_mean, dist_max])
r = md.predict([siga, sigb])
