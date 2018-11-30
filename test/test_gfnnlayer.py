

import tensorflow as tf
import numpy as np
from MSYNC import GFNN
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNN)

nosc = 128
batch_size = 1
Fs = 16000.0
dt = 1.0/Fs
t = np.arange(0, 0.1, dt)
ff = 800.0 # 2.095676
# sin = np.float32(0.2 * np.sin(2 * np.pi * ff * t))
sin = np.complex64(0.2 * np.exp(1j * 2 * np.pi * ff * t))
sin = np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0)

model = tf.keras.Sequential()
model.add(GFNN.GFNNLayer(nosc, dt, input_shape=(t.size,)))

r = model.predict(sin)
plt.imshow(r[0, :, :, 0])