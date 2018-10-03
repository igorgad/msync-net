
import GFNN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNN)


# Params
nosc = 128
batch_size = 2
boff = 100
Fs = 60.0
dt = 1.0/Fs
audio_file = '/media/igor/DATA/Dataset/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_02.wav'
samples_length = 1024
downsample_rate = 4

# Load Audio
audio_binary = tf.read_file(audio_file)
waveform = tf.reduce_mean(tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=44100, channel_count=2), axis=1)
waveform = tf.gather(waveform, tf.range(0, tf.size(waveform), downsample_rate), axis=0)
waveform = 0.25 * waveform / tf.reduce_max(waveform)
waveform = tf.contrib.signal.frame(waveform, samples_length, samples_length)

sin = tf.complex(waveform[boff:boff+batch_size, :], 0.0)
#sin = tf.placeholder(tf.float32, shape=[1, samples_length, 1])

with tf.device('/device:GPU:1'):
    gfnn = GFNN.GFNN(nosc, dt, use_hebbian_learning=False, avoid_nan=False)
    z_state, c_state = gfnn.gfnn(sin)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

sz, sc = sess.run([z_state, c_state])

sb = 1
freq = gfnn._f
xticks = np.arange(0, nosc, 50)
xfreq = freq[range(0, nosc, 50)]
xlabels = ["%.1f" % x for x in xfreq]

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.imshow(np.real(sz[0:sb,:,:].reshape([-1, nosc])).T, cmap='gray')
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax2.imshow(np.angle(sz[0:sb,:,:].reshape([-1, nosc])).T, cmap='gray')
ax2.set_yticks(xticks)
ax2.set_yticklabels(xlabels)

ax3.imshow(np.real(sc[0,-1,:,:]).T, cmap='gray')
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)
ax3.set_yticks(xticks)
ax3.set_yticklabels(xlabels)
