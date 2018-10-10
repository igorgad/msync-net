

import tensorflow as tf
import numpy as np
from models import simple_gfnn
from models import loss
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import importlib
importlib.reload(simple_gfnn)
importlib.reload(loss)

nosc = 180
batch_size = 128
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

callback_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs', batch_size=batch_size, write_images=True)
model = simple_gfnn.simple_gfnn_cca_v0(model_params)
model.fit([sin, sin], sin, validation_data=[[sin, sin], sin], validation_steps=1, epochs=2, steps_per_epoch=1, callbacks=[callback_tb])
