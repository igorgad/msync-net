
import tensorflow as tf
import numpy as np
from trainer.Model import MeanShift
import trainer.dataset_interface as dataset_interface
import matplotlib.pyplot as plt


mat = np.float32(np.random.rand(4, 2, 385))

ksig = tf.keras.Input(shape=(2, 385))
dist_mean = MeanShift(n=10, bw=20.0)(ksig)

md = tf.keras.Model(ksig, dist_mean)
md.summary()
r = md.predict(mat)