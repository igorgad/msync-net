

import tensorflow as tf
import numpy as np
from MSYNC import stft_model
from MSYNC import loss
from MSYNC import stats
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import importlib
importlib.reload(stft_model)
importlib.reload(loss)
importlib.reload(stats)

nosc = 180
batch_size = 128
Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 100, dt)
ff = 1.0 # 2.095676
# sin = np.float32(0.2 * np.sin(2 * np.pi * ff * t))
p1 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * ff * t[:t.size//2]))
p0 = np.complex64(0.2 * np.exp(1j * 2 * np.pi * 2.0 * t[t.size//2:]))
sin1 = np.repeat(np.expand_dims(np.concatenate([p1, p0], axis=0), 0),  batch_size, axis=0)
sin2 = np.repeat(np.expand_dims(np.concatenate([p0, p1], axis=0), 0),  batch_size, axis=0)

osc_params = {'f_min': 0.125,
                'f_max': 8.0,
                'alpha': -1.0,
                'beta1': -1.0,
                'beta2': 0.0,
                'delta1': 0.0,
                'delta2': 0.0,
                'eps': 1.0,
                'k': 1.0}

model_params = {'stft_frame_length': 512,
                'stft_frame_step': 256,
                'input_shape': (sin2.shape[-1],),
                'outdim_size': 128,
                'pre_train_lr': 0.001,
                'dctw_lr': 0.001,
                'v1_weights_file': './saved_models/v1_stft_weights.h5',
                'v2_weights_file': './saved_models/v2_stft_weights.h5',
                }

model, v1_model, v2_model = stft_model.simple_stft_cca(model_params)
model.compile(loss=loss.cca_loss, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['dctw_lr']))

tb = stats.TensorBoardDTW(log_dir='./logs/test_model', histogram_freq=1, batch_size=batch_size, write_images=True)
st = model.fit([sin1, sin2], sin1, validation_data=[[sin1, sin2], sin1], validation_steps=4, epochs=4, steps_per_epoch=2, callbacks=[tb])
