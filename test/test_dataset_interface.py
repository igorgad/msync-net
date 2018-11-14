
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
               'random_batch_size': 32,  # For training
               'sequential_batch_size': 16,  # For validation
               'repeat': 100000,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 4 * 15360
               }

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex = dataset_interface.pipeline(data_params)
ex = ex.make_one_shot_iterator().get_next()
r = sess.run(ex)

v1 = r[0]['v1input'][0].reshape(-1)
v2 = r[0]['v2input'][0].reshape(-1)
l = v1.size//2 + (np.argmax(r[1][0]) - data_params['sequential_batch_size']//2) * data_params['example_length']

plt.plot(v1)
plt.plot(v2)
plt.axvline(l)
plt.axvline(l + data_params['example_length'])
