
import tensorflow as tf
import numpy as np
from trainer.Model import EclDistanceMat, DiagMean
import trainer.dataset_interface
import matplotlib.pyplot as plt


sigb = np.float32(np.random.rand(4, 32, 64))
# siga = sigb + (4.0 * (1.0 - np.float32(np.random.rand(4, 32, 64))))
siga = np.concatenate([np.random.rand(4, 8, 64), sigb[:, :-8, :]], axis=1) + (4.0 * (1.0 - np.float32(np.random.rand(4, 32, 64))))

ksiga = tf.keras.Input(shape=(32, 64))
ksigb = tf.keras.Input(shape=(32, 64))

dist_mat = EclDistanceMat()([ksiga, ksigb])
dist_mean = DiagMean()(dist_mat)

md = tf.keras.Model([ksiga, ksigb], [dist_mat, dist_mean])
r = md.predict([siga, sigb])

plt.plot(r[1][0])
plt.figure()
plt.imshow(r[0][0, :, :, 0])
