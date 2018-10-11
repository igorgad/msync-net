
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import dataset_interface as dts
import MSYNC.simple_models as gfnn_model
import importlib
importlib.reload(dts)
importlib.reload(gfnn_model)


data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
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


