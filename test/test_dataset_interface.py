
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
import importlib
importlib.reload(dataset_interface)


data_root = '/media/igor/DATA/Dataset/BACH10/'
audio_dir = data_root + '/Audio/'
tfrecordfile = data_root + 'msync-bach10.tfrecord'
data_params = {'dataset_file': tfrecordfile,
               'dataset_root': audio_dir,
               'sample_rate': 44100//4}

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex = dataset_interface.pipeline(data_params)
r = sess.run(ex)
