
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
from MSYNC.Model import LogMel
import importlib
importlib.reload(dataset_interface)

data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 16000,
               'example_length': 15360,
               'batch_size': 4,
               'repeat': 100000,
               'shuffle_buffer': 128,
               'scale_value': 1.0,
               'max_delay': 15360 // 20
               }

data = dataset_interface.pipeline_v0(data_params)

logmel = LogMel(input_shape=(15360,))

model = tf.keras.Sequential()
model.add(logmel)

model.compile(loss='mse', optimizer='adam')
r = model.predict(data, steps=1)

plt.imshow(r[0])