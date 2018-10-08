
import tensorflow as tf
import numpy as np
import os


max_samples_delay = 1024
block_size = 2048
data_root = '/media/igor/DATA/Dataset/BACH10/'
audio_dir = data_root + '/Audio/'
tfrecordfile = data_root + 'msync-bach10.tfrecord'
train_test_ratio = 0.8


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


np.random.seed(0)
writer = tf.python_io.TFRecordWriter(tfrecordfile)

fs_data_root = os.fsencode(data_root)
fs_audio_dir = os.fsencode(audio_dir)

for folder in os.listdir(fs_audio_dir):
    files = sorted(os.listdir(fs_audio_dir + folder))
    instruments = []

    for file in files:
        instruments.append(file.split(b'.wav')[0].split(b'-')[-1])

    tf_example = tf.train.Example(features=tf.train.Features(feature={'folder': bytes_feature([folder]),
                                                                      'files': bytes_feature(files),
                                                                      'instruments': bytes_feature(instruments)
                                                                      }))

    writer.write(tf_example.SerializeToString())
    print('done folder ' + os.fsdecode(folder))

writer.close()
