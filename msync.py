
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import dataset_interface as dts
import models.simple_gfnn as gfnn_model
import importlib
importlib.reload(dts)
importlib.reload(gfnn_model)


data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 2048,
               'frame_step': 2048,
               'batch_size': 1,
               'repeat': 10,
               'shuffle': True
               }

model_params = {'num_osc': 180,
                'dt': 1/(44100//4),
                'input_shape': (2048,),
                'outdim_size': 128,
                'lr': 0.01
                }


data = dts.pipeline(data_params)
model = gfnn_model.simple_gfnn_cca_v0(model_params)

model.fit(data, epochs=2, steps_per_epoch=10)


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
