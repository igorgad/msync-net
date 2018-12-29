
import tensorflow as tf
import argparse
import numpy as np
import trainer.dataset_interface as dts
import matplotlib.pyplot as plt
import importlib
importlib.reload(dts)
import time

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB_v2.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 4 * 15360,
               'max_delay': 2 * 15360,
               'labels_precision': 15360 // 1,
               'random_batch_size': 16,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 1000,
               'from_bucket': False
               }

model_params = {'stft_window': 1600,
                'stft_step': 160,
                'num_mel_bins': 128,
                'num_spectrogram_bins': 1025,
                'lower_edge_hertz': 125.0,
                'upper_edge_hertz': 7500.0,
                'encoder_arch': 'lstm',
                'encoder_units': [512, 256],
                'top_units': [256, 128],
                'dropout': 0.5,
                'dmrn': False
                }

train_params = {'lr': 1.0e-4,
                'epochs': 40,
                'steps_per_epoch': 25,
                'val_steps': 25
                }

data_params['dataset_file'] = dataset_file
data_params['dataset_audio_dir'] = dataset_audio_root

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in train_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
params = parser.parse_known_args()[0]

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tfdataset = dts.cached_pipeline(params)
ex = tfdataset.make_one_shot_iterator().get_next()
rr = []

while True:
    try:
        r = sess.run(ex)
        rr.append(r['instruments'])

    except Exception as e:
        print (str(e))
        break

