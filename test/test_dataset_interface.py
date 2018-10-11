
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
import importlib
importlib.reload(dataset_interface)


data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 2048,
               'frame_step': 2048,
               'batch_size': 1,
               'repeat': 100,
               'shuffle_buffer': 128
               }

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex = dataset_interface.pipeline(data_params)
ex = ex.make_one_shot_iterator().get_next()
r = sess.run(ex)
