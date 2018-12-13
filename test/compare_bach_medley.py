
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import matplotlib.pyplot as plt
import importlib
importlib.reload(dts)
import time

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params_bach = {'sample_rate': 16000,
                     'example_length': 15360,  # almost 1 second of audio
                     'random_batch_size': 1,
                     'sequential_batch_size': 16,
                     'max_delay': 4 * 15360,
                     'instrument_1': 'bassoon',
                     'instrument_2': 'clarinet',
                     'split_seed': 2,
                     'split_rate': 0.7,
                     'debug_auto': True,
                     'scale_value': 1.0,
                     'shuffle_buffer': 32,
                     'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
                     'audio_root': './data/BACH10/Audio'
                     }

data_params_medley = {'sample_rate': 16000,
                     'example_length': 15360,  # almost 1 second of audio
                     'random_batch_size': 1,
                     'sequential_batch_size': 16,
                     'max_delay': 4 * 15360,
                     'instrument_1': 'electric bass',
                     'instrument_2': 'clean electric guitar',
                     'split_seed': 2,
                     'split_rate': 0.7,
                     'debug_auto': True,
                     'scale_value': 1.0,
                     'shuffle_buffer': 32,
                     'dataset_file': './data/MedleyDB/MSYNC-MedleyDB.tfrecord',
                     'audio_root': './data/MedleyDB/Audio'
                     }


config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

bach_dataset = dts.base_pipeline(data_params_bach).repeat(10).make_one_shot_iterator().get_next()
medl_dataset = dts.base_pipeline(data_params_medley).repeat(10).make_one_shot_iterator().get_next()
rb, rm = sess.run([bach_dataset, medl_dataset])
rrb = []
rrm = []

fig = plt.figure()
try:
    while True:
        rb, rm = sess.run([bach_dataset, medl_dataset])
        rrb.append(rb)
        rrm.append(rm)
except Exception as e:
    print (str(e))
    pass


for (rb, rm) in zip(rrb, rrm):
    fig.clf()
    axes = fig.subplots(2,2)
    axes[0, 0].plot(rb['signals'][0].reshape(-1))
    axes[1, 0].plot(rb['signals'][1].reshape(-1))
    axes[0, 1].plot(rm['signals'][0].reshape(-1))
    axes[1, 1].plot(rm['signals'][1].reshape(-1))

    axes[0,0].set_title('delay = ' + str(rb['delay'][1] - rb['delay'][0]))
    axes[0,1].set_title('delay = ' + str(rm['delay'][1] - rm['delay'][0]))

    fig.show()
    plt.pause(2)

