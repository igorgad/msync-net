
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import dataset_interface as dts
import models.simple_models as gfnn_model
import importlib
importlib.reload(dts)
importlib.reload(gfnn_model)


data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 1024,
               'frame_step': 1024,
               'batch_size': 1,
               'repeat': 100,
               'shuffle_buffer': 128
               }

model_params = {'num_osc': 360,
                'dt': 1/(44100//4),
                'input_shape': (1024,),
                'outdim_size': 128,
                'lr': 0.01
                }


data = dts.pipeline(data_params)
model = gfnn_model.simple_gfnn_cca_v0(model_params)

tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/v0', histogram_freq=1, batch_size=1, write_images=True)
model.fit(data, epochs=4, steps_per_epoch=5, validation_data=data, validation_steps=1, callbacks=[tb])


# Evaluate DTW
from dtw import dtw
from numpy.linalg import norm
import matplotlib.pyplot as plt

r = model.predict(data, steps=1)
r1 = r[:, :r.shape[1]//2]
r2 = r[:, r.shape[1]//2:]

dist, cost, acc_cost, path = dtw(r1.T, r2.T, dist=lambda x, y: norm(x - y, ord=1))
plt.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
