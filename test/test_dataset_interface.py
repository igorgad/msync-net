
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
               'example_length': 4 * 15360,  # almost 1 second of audio
               'random_batch_size': 128,
               'max_delay': 2 * 15360,
               'labels_precision': 15360 // 2,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 25,
               'stft_step': 160
               }

data_params['dataset_file'] = dataset_file
data_params['dataset_audio_dir'] = dataset_audio_root

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]
params = parser.parse_known_args()[0]

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tfdataset = dts.base_pipeline(params)
train_dataset = tfdataset.filter(dts.select_train_examples).prefetch(1)
val_dataset = tfdataset.filter(dts.select_val_examples).prefetch(1)

# train_dataset, val_dataset = dts.pipeline(data_params)

st = time.time()
rt = {'folder': [], 'delay': [], 'widx': []}
rv = {'folder': [], 'delay': [], 'widx': []}
num_epochs = 0
# fig = plt.figure()

while num_epochs < 1:
    try:
        num_ex_train = 0
        ex = train_dataset.make_one_shot_iterator().get_next()
        while num_ex_train < 100:
            r = sess.run(ex)
            rt['folder'].append(r['folder'])
            rt['delay'].append(r['delay'])
            rt['widx'].append(r['widx'])

            # fig.clf()
            # axes = fig.subplots(2, 1)
            # axes[0].plot(r['signals'][0].reshape(-1))
            # axes[1].plot(r['signals'][1].reshape(-1))
            # axes[0].set_title('delay = ' + str(r['delay'][1] - r['delay'][0]))
            #
            # fig.show()
            # plt.pause(2)
            num_ex_train += 1
    except Exception as e:
        # print (str(e))
        pass

    try:
        num_ex_val = 0
        ex = val_dataset.make_one_shot_iterator().get_next()
        while num_ex_val < 100:
            r = sess.run(ex)
            rv['folder'].append(r['folder'])
            rv['delay'].append(r['delay'])
            rv['widx'].append(r['widx'])
            num_ex_val += 1
    except Exception as e:
        # print(str(e))
        pass

    num_epochs += 1


print (time.time() - st)
t = num_ex_val + num_ex_train
print (str(t) + ', ' + str(num_ex_train / t) + ' | ' + str(num_ex_val / t))
