
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import matplotlib.pyplot as plt
import importlib
importlib.reload(dts)

dataset = 'medleydb'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,  # For training
               'sequential_batch_size': 8,  # For validation
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'drum set',         # Only valid for MedleyDB dataset
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'electric bass'  # Only valid for MedleyDB dataset
               }

data_params['dataset_file'] = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
data_params['audio_root'] = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
tfdataset = tfdataset.map(lambda ex: dts.parse_features_and_decode(ex, dts.features))
tfdataset = tfdataset.filter(lambda feat: dts.filter_instruments(feat, data_params))
tfdataset = tfdataset.map(lambda feat: dts.select_instruments(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.load_audio(feat, data_params), num_parallel_calls=4)
# tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
tfdataset = tfdataset.map(lambda feat: dts.compute_activations(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.mix_similar_instruments(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.scale_signals(feat, data_params), num_parallel_calls=4).cache()
tfdataset = tfdataset.map(lambda feat: dts.add_random_delay(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.frame_signals(feat, data_params), num_parallel_calls=4)
tfdataset = tfdataset.map(lambda feat: dts.sequential_batch(feat, data_params), num_parallel_calls=4)

train_dataset = tfdataset.filter(dts.select_train_examples).map(lambda feat: dts.prepare_examples(feat, data_params), num_parallel_calls=4)
val_dataset = tfdataset.filter(dts.select_val_examples).map(lambda feat: dts.prepare_examples(feat, data_params), num_parallel_calls=4)

try:
    num_ex_train = 0
    ex = train_dataset.make_one_shot_iterator().get_next()
    while True:
        r = sess.run(ex)
        # print (r)
        num_ex_train += 1
except Exception:
    pass

try:
    num_ex_val = 0
    ex = val_dataset.make_one_shot_iterator().get_next()
    while True:
        r = sess.run(ex)
        # print (r)
        num_ex_val += 1
except Exception:
    pass
