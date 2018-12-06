
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import matplotlib.pyplot as plt
from MSYNC.Model import LogMel


dataset = 'bach10'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 1,
               'sequential_batch_size': 1,
               'max_delay': 2,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 2,
               'split_rate': 0.8,
               'debug_auto': False
               }

# Get data pipelines
data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root
data = dts.base_pipeline(data_params).make_one_shot_iterator().get_next()

input = tf.keras.Input(shape=(data_params['example_length'],))
logmel = LogMel()(input)
model = tf.keras.Model(input, logmel)

sess = tf.keras.backend.get_session()
fig, (ax1, ax2) = plt.subplots(2,1)

while True:
    dt = sess.run(data)
    r1 = model.predict(dt['signals'][0])
    r2 = model.predict(dt['signals'][1])

    ax1.imshow(r1[0, :, :, 0], cmap='gray')
    ax2.imshow(r2[0, :, :, 0], cmap='gray')
    plt.pause(2)

