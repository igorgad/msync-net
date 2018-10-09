

import tensorflow as tf
import numpy as np
from models import GFNN
from models import loss
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(GFNN)

nosc = 180
batch_size = 512
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 20, dt)
ff = 1.0 # 2.095676
# sin = np.float32(0.2 * np.sin(2 * np.pi * ff * t))
sin1 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * ff * t[:t.size//2]))
sin0 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * 2.0 * t[t.size//2:]))
sin = np.concatenate([sin1, sin0], axis=0)
sin = np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0)


data_params = {'dataset_file': '/media/igor/DATA/Dataset/BACH10/msync-bach10.tfrecord',
               'audio_root': '/media/igor/DATA/Dataset/BACH10/Audio',
               'sample_rate': Fs,
               'frame_length': 2048,
               'frame_step': 1024,
               'batch_size': batch_size
               }

model_params = {'num_osc': nosc,
                'dt': dt,
                'input_shape': (sin.shape[1],),
                'outdim_size': 128,
                'lr': 0.01
                }

gfnn1 = GFNN.KerasLayer(model_params['num_osc'], model_params['dt'], input_shape=model_params['input_shape'])
model1 = tf.keras.Sequential()
model1.add(gfnn1)
model1.add(tf.keras.layers.LSTM(model_params['outdim_size']))

gfnn2 = GFNN.KerasLayer(model_params['num_osc'], model_params['dt'], input_shape=model_params['input_shape'])
model2 = tf.keras.Sequential()
model2.add(gfnn1)
model2.add(tf.keras.layers.LSTM(model_params['outdim_size']))

model = tf.keras.Sequential()
model.add(tf.keras.layers.concatenate([model1, model2]))

model_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['lr'])
model.compile(loss=loss.cca_loss, optimizer=model_optimizer)

model.fit(sin, sin)
