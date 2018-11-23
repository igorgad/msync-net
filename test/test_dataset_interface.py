
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
import time
import importlib
importlib.reload(dataset_interface)


dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,  # For training
               'sequential_batch_size': 8,  # For validation
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',         # Only valid for MedleyDB dataset
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',  # Only valid for MedleyDB dataset
               'debug_auto': False
               }


# Get data pipelines
data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 1
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex = dataset_interface.base_pipeline(data_params).repeat(4)
ex = ex.make_one_shot_iterator().get_next()

for i in range(40):
    r = sess.run(ex)

tm = []
for i in range(40):
    st = time.time()
    r = sess.run(ex)
    tm.append(time.time() - st)

tm = np.array(tm)
print(tm.mean())
