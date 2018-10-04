

import tensorflow as tf
import numpy as np
import GFNN
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNN)

nosc = 180
batch_size = 1
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 2, dt)
ff = 1.0 # 2.095676
sin = np.complex64(0.2 * np.exp(1j * 2 * np.pi * ff * t))
sin = np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0)

with tf.device('/device:GPU:1'):
    gfnn = GFNN.GFNN(nosc, dt, use_hebbian_learning=False, avoid_nan=False)
    z_state, c_state = gfnn.gfnn(sin)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

sz, sc = sess.run([z_state, c_state])

freq = gfnn._f
xticks = np.arange(0, nosc, 20)
xfreq = freq[range(0, nosc, 20)]
xlabels = ["%.1f" % x for x in xfreq]

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.imshow(np.abs(sz[0, :, :]).T, cmap='gray')
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax2.imshow(np.real(sz[0, :, :]).T, cmap='gray')
ax2.set_yticks(xticks)
ax2.set_yticklabels(xlabels)

ax3.plot(np.abs(sz[0, -200:, :]).mean(0))
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)

if gfnn._use_hebbian_learning:
    fig2, [ax1f2, ax2f2] = plt.subplots(1,2, figsize=(14, 8))
    ax1f2.imshow(np.real(sc[0, -1, :, :]).T, cmap='gray')
    ax1f2.set_xticks(xticks)
    ax1f2.set_xticklabels(xlabels)
    ax1f2.set_yticks(xticks)
    ax1f2.set_yticklabels(xlabels)

    ax2f2.imshow(np.angle(sc[0, -1, :, :]).T, cmap='gray')
    ax2f2.set_xticks(xticks)
    ax2f2.set_xticklabels(xlabels)
    ax2f2.set_yticks(xticks)
    ax2f2.set_yticklabels(xlabels)
