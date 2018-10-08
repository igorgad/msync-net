

import tensorflow as tf
import numpy as np
from models import GFNN
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNN)

nosc = 180
batch_size = 1
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 20, dt)
ff = 1.0 # 2.095676
# sin = np.float32(0.2 * np.sin(2 * np.pi * ff * t))
sin1 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * ff * t[:t.size//2]))
sin0 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * 2.0 * t[t.size//2:]))
sin = np.concatenate([sin1, sin0], axis=0)
sin = np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0)

inputs = tf.keras.Input((sin.shape[1],), dtype=tf.complex64)
gfnn_layer = GFNN.KerasLayer(nosc, dt, use_hebbian_learning=False)
out = gfnn_layer(inputs)

model = tf.keras.Model(inputs=inputs, outputs=out)
sz = model.predict(sin, batch_size=batch_size)

freq = gfnn_layer.gfnn._f
xticks = np.arange(0, nosc, 20)
xfreq = freq[range(0, nosc, 20)]
xlabels = ["%.1f" % x for x in xfreq]

fig, [ax1, ax3] = plt.subplots(2, figsize=(14, 8))
ax1.imshow(sz[0, :, :], cmap='gray')
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax3.plot(sz[0, :, -200:].mean(1))
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)
