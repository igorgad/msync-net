
import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wave


data_root = './data/BACH10/'
audio_dir = data_root + '/Audio/'
tfrecordfile = data_root + 'MSYNC-bach10.tfrecord'
train_test_ratio = 0.8

strings = [b'violin']
brass = [b'saxphone']
woods = [b'bassoon', b'clarinet']
tps = {'strings': strings, 'brass': brass, 'woods': woods}


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
    signals = []
    is_train = np.random.randint(100) < train_test_ratio * 100

    for file in files:
        instruments.append(file.split(b'.wav')[0].split(b'-')[-1])
        signals.append(np.float32(wave.read(fs_audio_dir + folder + b'/' + file)[1]).tostring())

    types = [os.fsencode(list(tps.keys())[np.nonzero([s.count(inst) for s in tps.values()])[0][0]]) for inst in instruments]

    tf_example = tf.train.Example(features=tf.train.Features(feature={'folder': bytes_feature([folder]),
                                                                      'is_train': int64_feature(np.int64(is_train)),
                                                                      'files': bytes_feature(files),
                                                                      'instruments': bytes_feature(instruments),
                                                                      'types': bytes_feature(types),
                                                                      'signals': bytes_feature(signals)
                                                                      }))

    writer.write(tf_example.SerializeToString())
    print('done folder ' + os.fsdecode(folder))

writer.close()
