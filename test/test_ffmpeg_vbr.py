
import numpy as np
import os
import ffmpeg
import matplotlib.pyplot as plt

audio_file = '/home/igor/DATA_DISK/Dataset/BACH10/Audio/01-AchGottundHerr/01-AchGottundHerr-violin.wav'
block_size = 1024

data, _ = ffmpeg.input(audio_file).output('pipe:', format='flac', acodec='flac', frame_size=block_size, ac=1, ar=16000).run(capture_stdout=True)
probe = ffmpeg.probe_bytes(data)

vbr = np.array([int(frame['pkt_size']) for frame in probe['frames']])
plt.plot(vbr)
