
import tensorflow as tf
import numpy as np
import os

features = {
        'folder': tf.VarLenFeature(tf.string),
        'files': tf.VarLenFeature(tf.string),
        'instruments': tf.VarLenFeature(tf.string)
}


def parse_features_and_decode(tf_example):
    parsed_features = tf.parse_single_example(tf_example, features)
    parsed_features['folder'] = tf.sparse_tensor_to_dense(parsed_features['folder'], b'')[0]
    parsed_features['files'] = tf.sparse_tensor_to_dense(parsed_features['files'], b'')
    parsed_features['instruments'] = tf.sparse_tensor_to_dense(parsed_features['instruments'], b'')
    return parsed_features


def load_audio(parsed_features, data_params):
    def load_file(file):
        audio_binary = tf.read_file(os.fsencode(data_params['dataset_root']) + b'/' + parsed_features['folder'] + b'/' + file)
        smp = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=data_params['sample_rate'], channel_count=1)
        return smp

    parsed_features['samples'] = tf.map_fn(load_file, parsed_features['files'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(parse_features_and_decode)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params))
    return tfdataset.make_one_shot_iterator().get_next()
