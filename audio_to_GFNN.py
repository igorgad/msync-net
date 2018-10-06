

import GFNN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(GFNN)




# Params
downsample_rate = 4
nosc = 180
batch_size = 1
boff = 100
Fs = 44100.0 / downsample_rate
dt = 1.0/Fs
# audio_file = '/media/igor/DATA/Dataset/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_05.wav'
audio_file = '/home/pepeu/workspace/Dataset/Audio/AClassicEducation_NightOwl_STEMS/AClassicEducation_NightOwl_STEM_05.wav'
samples_length = 1024 * 8

# Load Audio
audio_binary = tf.read_file(audio_file)
waveform = tf.reduce_mean(tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=np.int32(Fs), channel_count=2), axis=1)
# waveform = tf.gather(waveform, tf.range(0, tf.size(waveform), downsample_rate), axis=0)
waveform = 0.25 * waveform / tf.reduce_max(waveform)

waveform = tf.contrib.signal.frame(waveform, samples_length, samples_length)
sin = tf.complex(waveform[boff:boff+batch_size, :], 0.0)

with tf.device('/device:GPU:1'):
    gfnn = GFNN.GFNN(nosc, dt, use_hebbian_learning=False)
    z_state, c_state = gfnn.run(sin)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

wav, sz, sc = sess.run([sin, z_state, c_state])

freq = gfnn._f
xticks = np.arange(0, nosc, 20)
xfreq = freq[range(0, nosc, 20)]
xlabels = ["%.1f" % x for x in xfreq]

sb = 0
in_stim = np.real(wav[sb,:])
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, figsize=(14, 8))
ax1.plot(in_stim)
ax1.set_title('Input Stimulus')

ax2.imshow(np.abs(sz[sb, :, :]), cmap='gray')
ax2.set_yticks(xticks)
ax2.set_yticklabels(xlabels)
ax2.set_title('Oscillators Amplitude Response')

ax3.imshow(np.real(sz[sb, :, :]), cmap='gray')
ax3.set_yticks(xticks)
ax3.set_yticklabels(xlabels)
ax3.set_title('Oscillators Phase Response')

ax4.plot(np.abs(sz[sb, :, -200:]).mean(0))
ax4.set_xticks(xticks)
ax4.set_xticklabels(xlabels)
ax4.set_title('Average Absolute Response')

# import sounddevice as sd
# sd.play(in_stim, Fs)


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
