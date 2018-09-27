
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GFNNCell

# Params
nosc = 360
omega = 2 * np.pi * np.logspace(0, 1, nosc, dtype=np.float32) / 10
batch_size = 64
boff = 100
Fs = 30.0
dt = 1.0/Fs
audio_file = '/media/igor/DATA/Dataset/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_03.wav'
samples_length = 1024

# Load Audio
audio_binary = tf.read_file(audio_file)
waveform = tf.reduce_mean(tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=44100, channel_count=2), axis=1)
waveform = tf.contrib.signal.frame(waveform, samples_length, samples_length)

sin = tf.expand_dims(waveform[boff:boff+batch_size, :], axis=-1)
#sin = tf.placeholder(tf.float32, shape=[1, samples_length, 1])

with tf.device('/device:GPU:1'):
    gfnn_cell = GFNNCell.GFNNCell(nosc, dt, use_hebbian_learning=False)
    initial_state = gfnn_cell.zero_state(batch_size, tf.complex64)
    outputs, state = tf.nn.dynamic_rnn(gfnn_cell, sin, initial_state=initial_state)

# Run
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


r, s = sess.run([outputs, state])


sr = np.array(r).reshape([-1, nosc])
sc = np.array(s.c[0,:]).reshape([nosc, nosc])

freq = omega / (2*np.pi*dt)
xticks = np.arange(0, nosc, 50)
xfreq = freq[range(0, nosc, 50)]
xlabels = ["%.1f" % x for x in xfreq]

plt.figure()
plt.imshow(np.abs(sr)[0:samples_length,:])
plt.xticks(xticks, xlabels)

plt.figure()
plt.imshow(np.abs(sc))
plt.xticks(xticks, xlabels)
