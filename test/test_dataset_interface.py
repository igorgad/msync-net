
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

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,
               'sequential_batch_size': 8,
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 2,
               'num_folds': 5,
               'debug_auto': False
               }

data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tfdataset = dts.base_pipeline(data_params)
train_dataset = tfdataset.filter(lambda feat: dts.select_folds(feat, np.arange(data_params['num_folds'] - 1, dtype=np.int32))).prefetch(32)
val_dataset = tfdataset.filter(lambda feat: dts.select_folds(feat, [data_params['num_folds'] - 1])).prefetch(32)

# train_dataset, val_dataset = dts.pipeline(data_params)
# ex = train_dataset.make_one_shot_iterator().get_next()
# stft = tf.pow(tf.abs(tf.contrib.signal.stft(ex['signals'], 1600, 160, pad_end=True)), 2)

st = time.time()
rt = []
rv = []
try:
    num_ex_train = 0
    ex = train_dataset.make_one_shot_iterator().get_next()
    while True:
        r = sess.run(ex)
        # plt.clf()
        # fig, [ax1, ax2] = plt.subplots(2,1)
        # ax1.plot(r['signals'][0].reshape(-1))
        # ax2.plot(r['signals'][1].reshape(-1))
        # plt.title('delay = ' + str(r['delay'][1] - r['delay'][0]))
        num_ex_train += 1
except Exception as e:
    # print (str(e))
    pass

try:
    num_ex_val = 0
    ex = val_dataset.make_one_shot_iterator().get_next()
    while True:
        r = sess.run(ex)
        rv.append(r['folder'])
        # plt.clf()
        # plt.plot(r[0]['v1input'].reshape(-1))
        # plt.plot(r[0]['v2input'].reshape(-1))
        # plt.title('delay = ' + str(data_params['example_length'] * np.nonzero(r[1])[0][0]))
        plt.pause(0.1)
        num_ex_val += 1
except Exception as e:
    # print(str(e))
    pass

print (time.time() - st)
t = num_ex_val + num_ex_train
print (str(t) + ', ' + str(num_ex_train / t) + ' | ' + str(num_ex_val / t))
