
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
import importlib
importlib.reload(dataset_interface)


data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//8,
               'example_length': 20480,
               'batch_size': 8,
               'repeat': 100000,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 20480 // 20
               }

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex = dataset_interface.base_pipeline(data_params)
# ex = dataset_interface.pipeline(data_params)
ex = ex.make_one_shot_iterator().get_next()
r = sess.run(ex)
