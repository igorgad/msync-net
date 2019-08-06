
import os
import tensorflow as tf
import numpy as np
import argparse
import trainer.dataset_interface as dataset_interface
import trainer.Model as modules
from dtw import dtw
from numpy.linalg import norm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB_v2.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 4 * 15360,
               'num_examples': 1,
               'num_examples_test': 1,
               'max_delay': 2 * 15360,
               'labels_precision': 15360 // 2,
               'random_batch_size': 1,
               'test_batch_size': 1,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 0,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 1000,
               'from_bucket': False,
               'num_folds': 5,
               'bw': 1.0
               }

model_params = {'stft_window': 3200,
                'stft_step': 160,
                'num_mel_bins': 256,
                'num_spectrogram_bins': 2049,
                'lower_edge_hertz': 125.0,
                'upper_edge_hertz': 7500.0
                }

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
parser.add_argument('--logdir', type=str, default='logs/', help='The directory to store the experiments logs (default: logs/)')
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--dataset_audio_dir', type=str, default=dataset_audio_root, help='Directory to fetch wav files (default: %s)' % str(dataset_audio_root))
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]

params = parser.parse_known_args()[0]

tf.set_random_seed(26)

_, __, test_data = dataset_interface.kfold_pipeline(params, train_folds=[0], val_folds=[0], test_folds=[0, 1, 2, 3, 4])
test_example = test_data.make_one_shot_iterator().get_next()
v1_input, v2_input = tf.unstack(test_example[0]['inputs'], axis=-1)
v1_stft_t = tf.squeeze(modules.LogMel(params)(v1_input), axis=0)
v2_stft_t = tf.squeeze(modules.LogMel(params)(v2_input), axis=0)
label_t = tf.argmax(tf.squeeze(test_example[1], axis=0))

sess = tf.keras.backend.get_session()

num_examples_correct = 0
total_num_examples = 1000
preds = []
preds_path = []
labels = []

for step in range(total_num_examples):
    print ('step %d' % step)
    v1_feat, v2_feat, label = sess.run([v1_stft_t, v2_stft_t, label_t])
    dist, cost, acc_cost, path = dtw(v2_feat, v1_feat, dist=lambda x, y: norm(x - y, ord=1))
    dmes = np.concatenate([acc_cost[-1, acc_cost.shape[1]//2:], np.flip(acc_cost[acc_cost.shape[0]//2:, -1])], axis=0)
    pred_label = np.argmax(dmes) - acc_cost.shape[0]//2
    pred_path = np.median(path[0] - path[1])
    label = label - acc_cost.shape[0]//2
    preds.append(pred_label)
    preds_path.append(pred_path)
    labels.append(label)


preds = np.array(preds)
preds_path = np.array(preds_path)
labels = np.array(labels)

acc96 = (preds - labels <= 48).astype(np.float32).mean()
acc48 = (preds - labels <= 24).astype(np.float32).mean()
acc24 = (preds - labels <= 12).astype(np.float32).mean()
print ('pred_labels: ' + str([acc96, acc48, acc24]))

acc96 = (preds_path - labels <= 48).astype(np.float32).mean()
acc48 = (preds_path - labels <= 24).astype(np.float32).mean()
acc24 = (preds_path - labels <= 12).astype(np.float32).mean()
print ('pred_path: ' + str([acc96, acc48, acc24]))