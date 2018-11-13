

import tensorflow as tf
import numpy as np
from MSYNC.Model import LogMel, EclDistanceMat, DiagMean
import dataset_interface
import matplotlib.pyplot as plt


siga = np.float32(np.random.rand(4, 32, 15360))
sigb = siga  #np.float32(np.random.rand(4, 32, 64))

ksiga = tf.keras.Input(shape=(32, 15360))
ksigb = tf.keras.Input(shape=(32, 15360))

logmel1 = tf.keras.layers.TimeDistributed(LogMel())(ksiga)
logmel2 = tf.keras.layers.TimeDistributed(LogMel())(ksigb)

flat1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(logmel1)
flat2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(logmel2)

dense1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))(flat1)
dense2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))(flat2)

ecl_mat = EclDistanceMat()([dense1, dense2])

model1 = tf.keras.Model([ksiga, ksigb], ecl_mat)
