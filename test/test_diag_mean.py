
import tensorflow as tf
import numpy as np
from MSYNC.Model import DiagMean
import dataset_interface
import matplotlib.pyplot as plt


mat = np.float32(np.random.rand(4, 32, 32))

ksig = tf.keras.Input(shape=(32, 32))
dist_mean = DiagMean()(ksig)

md = tf.keras.Model(ksig, dist_mean)
r = md.predict(mat)