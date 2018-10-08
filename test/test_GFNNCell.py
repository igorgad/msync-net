from test import GFNNCell
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNNCell)

nosc = 180
batch_size = 1
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 20, dt)
ff = 2
sin = 0.01 * np.sin(t*2*np.pi*ff)
sin = np.float32(np.expand_dims(np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0), axis=-1))
#sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*1*np.pi),  np.zeros(t.size//2)])

with tf.device('/device:GPU:0'):
    gfnn_cell = GFNNCell.GFNNCell(nosc, dt, use_hebbian_learning=False, avoid_nan=False)
    initial_state = gfnn_cell.zero_state(batch_size, dtype=tf.complex64)
    outputs, state = tf.nn.dynamic_rnn(gfnn_cell, sin, initial_state=initial_state)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

r, s = sess.run([outputs, state])

sr = np.array(r).reshape([-1, nosc])
sc = np.array(s.c[0,:]).reshape([nosc, nosc])

fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
freq = gfnn_cell._f
xticks = np.arange(0, nosc, 50)
xfreq = freq[range(0, nosc, 50)]
xlabels = ["%.1f" % x for x in xfreq]

ax1.imshow(np.real(sr).T)
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax2.plot(np.real(sr).mean(0))
ax2.set_xticks(xticks)
ax2.set_xticklabels(xlabels)

ax3.imshow(np.real(sc))
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)
ax3.set_yticks(xticks)
ax3.set_yticklabels(xlabels)

