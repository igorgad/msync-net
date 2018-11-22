
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset_interface as dts
from MSYNC.GFNN import GFNN
from MSYNC.GFNN import HebCon

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = 'bach10'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 2048,  # almost 1 second of audio
               'random_batch_size': 1,  # For training
               'sequential_batch_size': 1,  # For validation
               'max_delay': 512,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',         # Only valid for MedleyDB dataset
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',  # Only valid for MedleyDB dataset
               'debug_auto': True
               }

logname = 'gfnn-' + dataset + ''.join(['-%s=%s' % (key, str(value).replace(' ', '_')) for (key, value) in data_params.items()])

data_params['scale_value'] = 0.2
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root
train_data = dts.base_pipeline(data_params)
ex = train_data.make_one_shot_iterator().get_next()

with tf.device('/device:GPU:0'):
    vec1 = tf.complex(tf.reshape(ex['signals'][0], [data_params['random_batch_size'], -1]), 0.0)
    vec2 = tf.complex(tf.reshape(ex['signals'][1], [data_params['random_batch_size'], -1]), 0.0)

    gfnn1 = GFNN(64, 1/data_params['sample_rate'])
    gfnn2 = GFNN(64, 1/data_params['sample_rate'])
    hebcon = HebCon(1/data_params['sample_rate'])

    rg1 = gfnn1.run(vec1)[0]
    rg2 = gfnn1.run(vec2)[0]
    rheb = hebcon.run(rg1, rg2)


config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

ex, r1, r2, rh = sess.run([ex, rg1, rg2, rheb])

plt.imshow(np.abs(rh[0,:,:,-1]))
