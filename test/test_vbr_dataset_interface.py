
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
               'num_examples': 1,
               'num_examples_test': 3,
               'max_delay': 2 * 15360,
               'labels_precision': 0,
               'random_batch_size': 16,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 1000,
               'from_bucket': False,
               'block_size': 1024
               }

model_params = {'encoder_units': [512, 256],
                'top_units': [256, 128],
                'dropout': 0.5,
                'dmrn': False,
                'residual_connection': False,
                'culstm': True
                }

train_params = {'lr': 1.0e-4,
                'epochs': 50,
                'steps_per_epoch': 25,
                'val_steps': 25,
                'metrics_range': [15360 // 1, 15360 // 2, 15360 // 4],
                'verbose': 1,
                'num_folds': 5
                }

data_params['dataset_file'] = dataset_file
data_params['dataset_audio_dir'] = dataset_audio_root

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in train_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
data_params = parser.parse_known_args()[0]

tfdataset = tf.data.TFRecordDataset(data_params.dataset_file)
tfdataset = tfdataset.map(lambda ex: dts.parse_features_and_decode(ex, dts.features))
tfdataset = tfdataset.filter(lambda feat: dts.filter_instruments(feat, data_params))
tfdataset = tfdataset.map(lambda feat: dts.select_instruments(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.load_vbr(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.scale_signals(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.compute_activations_vbr(feat, data_params), num_parallel_calls=4)
ex = tfdataset.make_one_shot_iterator().get_next()

sess = tf.Session()
r = sess.run(ex)
