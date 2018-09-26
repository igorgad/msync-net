
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GFNNCell

# Params
nosc = 360
batch_size = 256
boff = 1000
Fs = 60.0
dt = 1.0/Fs
audio_file = '/media/igor/DATA/Dataset/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_01.wav'
samples_length = 1024

# Load Audio
audio_binary = tf.read_file(audio_file)
waveform = tf.reduce_mean(tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=44100, channel_count=2), axis=1)
waveform = tf.contrib.signal.frame(waveform, samples_length, samples_length)

sin = tf.expand_dims(waveform[boff:boff+batch_size, :], axis=-1)

with tf.device('/device:GPU:1'):
    gfnn_cell = GFNNCell.GFNNCell(nosc, dt, use_hebbian_learning=True)
    initial_state = gfnn_cell.zero_state(batch_size, tf.complex64)
    outputs, state = tf.nn.dynamic_rnn(gfnn_cell, sin, initial_state=initial_state)

# Run
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

r, s = sess.run([outputs, state])


sr = np.array(r).sum(axis=2).reshape([-1])
sc = np.array(s.c[0,:]).reshape([nosc, nosc])

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
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
