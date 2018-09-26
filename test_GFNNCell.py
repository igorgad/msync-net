
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GFNNCell

nosc = 360
batch_size = 1
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 20, dt)
ff = 2
sin = 0.25 * np.sin(t*2*np.pi*ff)
sin = np.float32(np.expand_dims(np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0), axis=-1))
#sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*1*np.pi),  np.zeros(t.size//2)])

with tf.device('/device:GPU:1'):
    gfnn_cell = GFNNCell.GFNNCell(nosc, dt)
    initial_state = gfnn_cell.zero_state(batch_size, tf.complex64)
    outputs, state = tf.nn.dynamic_rnn(gfnn_cell, sin, initial_state=initial_state)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

r, s = sess.run([outputs, state])


sr = np.array(r).sum(axis=2)[0,:]
sc = np.array(s.c[0,:]).reshape([nosc, nosc])

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.plot(sin[0,:,0])
ax1.plot(sr)

n = sr.size
T = n * dt
freq = np.arange(n)[1:n//2]/T
logfreq = np.log(freq)
xticks = np.linspace(logfreq[0], logfreq[-1], 10)

ffsr = 1/n * np.abs(np.fft.fft(sr)[1:n//2])
ax2.plot(logfreq, np.log(ffsr))
ax2.set_xticks(xticks)
ax2.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])

ax3.imshow(np.abs(sc))