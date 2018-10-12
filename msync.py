
import os
import dataset_interface as dts
import MSYNC.simple_models as gfnn_model
import MSYNC.stats as stats
import importlib

os.environ["CUDA_VISIBLE_DEVICES"]="0"
importlib.reload(dts)
importlib.reload(gfnn_model)
importlib.reload(stats)

osc_params = {'f_min': 20.0,
              'f_max': 5000.0,
              'alpha': -1.0,
              'beta1': -1.0,
              'beta2': 0.0,
              'delta1': 0.0,
              'delta2': 0.0,
              'eps': 1.0,
              'k': 1.0
              }

model_params = {'num_osc': 360,
                'dt': 1/(44100//4),
                'osc_params': osc_params,
                'input_shape': (1024,),
                'outdim_size': 128,
                'lr': 0.01
                }

data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 1024,
               'frame_step': 1024,
               'batch_size': 1,
               'repeat': 100,
               'shuffle_buffer': 128
               }

data = dts.pipeline(data_params)
model = gfnn_model.simple_gfnn_cca_v0(model_params)

tb = stats.TensorBoardDTW(log_dir='./logs/v0', histogram_freq=1, batch_size=data_params['batch_size'], write_images=True)
model.fit(data, epochs=4, steps_per_epoch=5, validation_data=data, validation_steps=1, callbacks=[tb])
