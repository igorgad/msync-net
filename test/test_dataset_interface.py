
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
import importlib
importlib.reload(dataset_interface)


data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 32,
               'sequential_batch_size': 16,
               'repeat': 100000,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 4 * 15360
               }

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# ex = dataset_interface.train_pipeline(data_params)
ex = dataset_interface.test_pipeline(data_params)
ex = ex.make_one_shot_iterator().get_next()
r = sess.run(ex)
