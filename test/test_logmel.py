
import tensorflow as tf
import numpy as np
import dataset_interface
import matplotlib.pyplot as plt
from MSYNC.Model import logmel_func
import importlib
importlib.reload(dataset_interface)

data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 16000,
               'example_length': 15360,
               'batch_size': 64,
               'repeat': 100000,
               'shuffle_buffer': 4,
               'scale_value': 1.0,
               'max_delay': 15360 // 20
               }


data = dataset_interface.pipeline(data_params)
ex = data.make_one_shot_iterator().get_next()
inputs = ex[0]['v1input']

# inputs = np.float32(np.random.rand(1024, 15360))

mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.float32,
        name=None
    )


stft = tf.abs(tf.contrib.signal.stft(inputs, 400, 160, pad_end=True))
mel = tf.tensordot(stft, mel_matrix, 1)
mel.set_shape(stft.shape[:-1].concatenate(mel_matrix.shape[-1:]))
mel_log = tf.log(mel + 0.01)
mel_log = tf.expand_dims(mel_log, -1)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

for i in range(1000):
    r = sess.run(mel_log)
    print ('run ' + str(i))
