
import tensorflow as tf
import numpy as np
import yaml
import os

# Params
data_root = './data/MedleyDB/'
audio_dir = data_root + '/Audio/'
activation_dir = data_root + '/ACTIVATION_CONF/'
metadata_dir = data_root + '/METADATA/'
tfrecordfile = data_root + 'MSYNC-MedleyDB_v2.tfrecord'
train_test_ratio = 0.8


#### Dataset type classification
rythm = ['gong', 'auxiliary percussion', 'bass drum', 'bongo', 'chimes', 'claps', 'cymbal', 'drum machine', 'darbuka', 'glockenspiel', 'doumbek', 'drum set', 'kick drum', 'shaker', 'snare drum',
         'tabla', 'tambourine', 'timpani', 'toms', 'vibraphone', 'high hat', 'castanet']
eletronic = ['Main System', 'fx/processed sound', 'sampler', 'scratches']
strings = ['gu', 'zhongruan', 'liuqin', 'guzheng', 'erhu', 'harp', 'electric bass', 'acoustic guitar', 'banjo', 'cello', 'cello section', 'clean electric guitar', 'distorted electric guitar',
           'double bass', 'lap steel guitar', 'mandolin', 'string section', 'viola', 'viola section', 'violin', 'violin section', 'yangqin', 'zhongruan', 'dilruba', 'sitar']
brass = ['piccolo', 'soprano saxophone', 'horn section', 'alto saxophone', 'bamboo flute', 'baritone saxophone', 'bass clarinet', 'bassoon', 'brass section', 'clarinet', 'clarinet section', 'dizi',
         'flute', 'flute section', 'french horn', 'french horn section', 'oboe', 'oud', 'tenor saxophone', 'trombone', 'trombone section', 'trumpet', 'trumpet section', 'tuba']
voice = ['female singer', 'male rapper', 'male singer', 'male speaker', 'vocalists', 'female speaker',  'male screamer']
melody = ['electric piano', 'accordion', 'piano', 'synthesizer', 'tack piano', 'harmonica', 'melodica', 'electronic organ']
unknow = ['Unlabeled', 'crowd']
tps = {'rythm': rythm, 'electronic': eletronic, 'strings': strings, 'brass': brass, 'voice': voice, 'melody': melody, 'unknow': unknow}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


np.random.seed(2)
# options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter(tfrecordfile)

fs_data_root = os.fsencode(data_root)
fs_audio_dir = os.fsencode(audio_dir)
fs_activation_dir = os.fsencode(activation_dir)
fs_metadata_dir = os.fsencode(metadata_dir)

for file in sorted(os.listdir(fs_metadata_dir)):
    yml = yaml.load(open(os.path.join(fs_metadata_dir, file), 'r').read(-1))

    try:
        lab = open(os.path.join(fs_activation_dir, file.split(b'_METADATA.yaml')[0] + b'_ACTIVATION_CONF.lab'), 'r').read(-1)
        labmat = np.stack([np.fromstring(lb, dtype=np.float32, sep=',') for lb in lab.split('\n')[1:-1]]).transpose()
        lablist = [lab.tostring() for lab in labmat]
    except:
        lablist = []

    folder =  os.fsencode(yml['stem_dir'])
    files = [os.fsencode(yml['stems']['S%.2d' % s]['filename']) for s in range(1, len(yml['stems']) + 1)]
    instruments = [os.fsencode(yml['stems']['S%.2d' % s]['instrument'] if type(yml['stems']['S%.2d' % s]['instrument']) == str else yml['stems']['S%.2d' % s]['instrument'][0]) for s in range(1, len(yml['stems']) + 1)]
    types = [os.fsencode(list(tps.keys())[np.nonzero([s.count(yml['stems']['S%.2d' % si]['instrument'] if type(yml['stems']['S%.2d' % si]['instrument']) == str else yml['stems']['S%.2d' % si]['instrument'][0]) for s in tps.values()])[0][0]]) for si in range(1, len(yml['stems']) + 1)]
    is_train = np.random.randint(100) < train_test_ratio * 100

    tf_example = tf.train.Example(features=tf.train.Features(feature={'folder': bytes_feature([folder]),
                                                                      'is_train': int64_feature(np.int64(is_train)),
                                                                      'files': bytes_feature(files),
                                                                      'instruments': bytes_feature(instruments),
                                                                      'types': bytes_feature(types),
                                                                      'activations': bytes_feature(lablist)
                                                                      }))

    writer.write(tf_example.SerializeToString())
    print('################################ processed data from ' + yml['mix_filename'])

writer.close()
