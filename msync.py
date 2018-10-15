
import os
import tensorflow as tf
import dataset_interface as dts
import MSYNC.simple_models as gfnn_model
import MSYNC.stats as stats
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
importlib.reload(dts)
importlib.reload(gfnn_model)
importlib.reload(stats)

osc_params = {'f_min': 20.0,
              'f_max': 5000.0,
              'alpha': -1.0,
              'beta1': -10.0,
              'beta2': 0.0,
              'delta1': 0.0,
              'delta2': 0.0,
              'eps': 1.0,
              'k': 1.0
              }

model_params = {'num_osc': 256,
                'dt': 1/(44100//4),
                'osc_params': osc_params,
                'input_shape': (2048,),
                'outdim_size': 128,
                'lr': 0.01
                }

data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 2048,
               'frame_step': 1024,
               'batch_size': 1,
               'repeat': 10,
               'shuffle_buffer': 32,
               'scale_value': 0.25
               }

# Get models
dctw_model, v1_model, v2_model = gfnn_model.simple_gfnn_cca_v0(model_params)

# Apply denoising autoencoder pre-training
v1_data = dts.v1_pipeline(data_params)
v1_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/v0/pre-train1')
v1_model.fit(v1_data, epochs=4, steps_per_epoch=5, callbacks=[v1_tb])

v2_data = dts.v2_pipeline(data_params)
v2_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/v0/pre-train2')
v2_model.fit(v2_data, epochs=4, steps_per_epoch=5, callbacks=[v2_tb])

# DCTW Training
dctw_data = dts.dctw_pipeline(data_params)
dctw_tb = stats.TensorBoardDTW(log_dir='./logs/v0/dctw', histogram_freq=1, batch_size=data_params['batch_size'], write_images=True)
dctw_model.fit(dctw_data, epochs=4, steps_per_epoch=5, validation_data=dctw_data, validation_steps=1, callbacks=[dctw_tb])
