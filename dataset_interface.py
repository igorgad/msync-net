
import tensorflow as tf
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
        audio_binary = tf.read_file(os.fsencode(data_params['audio_root']) + b'/' + parsed_features['folder'] + b'/' + file)
        smp = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=data_params['sample_rate'], channel_count=1)
        return smp[:, 0]

    parsed_features['signals'] = tf.map_fn(load_file, parsed_features['files'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def copy_v0_to_vall(parsed_features):
    num_signals = tf.shape(parsed_features['signals'])[0]
    parsed_features['signals'] = tf.map_fn(lambda sig: parsed_features['signals'][0], tf.range(num_signals), dtype=tf.float32, infer_shape=False)
    return parsed_features


def scale_signals(parsed_features, data_params):
    def scale_signal(signal):
        sig = 2 * data_params['scale_value'] * (signal - tf.reduce_min(signal)) / (tf.reduce_max(signal) - tf.reduce_min(signal)) - data_params['scale_value']
        return sig

    parsed_features['signals'] = tf.map_fn(scale_signal, parsed_features['signals'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def add_random_delay(parsed_features, data_params):
    num_signals = tf.shape(parsed_features['signals'])[0]
    delay = tf.random_uniform([num_signals], 1, data_params['max_delay'], dtype=tf.int32)

    def add_delay(signal_index):
        return tf.concat([tf.zeros(delay[signal_index], dtype=tf.float32), parsed_features['signals'][signal_index, 0:-delay[signal_index]]], axis=0)

    parsed_features['delay'] = delay
    parsed_features['signals'] = tf.map_fn(add_delay, tf.range(num_signals), dtype=tf.float32, infer_shape=False)
    return parsed_features


def limit_amount_of_samples(parsed_features, data_params):
    nws_per_signal = tf.map_fn(lambda sig: tf.shape(sig)[0], parsed_features['signals'], dtype=tf.int32, infer_shape=False)
    nws_min = tf.reduce_min(nws_per_signal)
    min_range = tf.where(data_params['example_length'] > nws_min, 0, tf.random_uniform([1], 0, tf.abs(nws_min - data_params['example_length']), dtype=tf.int32)[0])
    max_range = tf.where(data_params['example_length'] > nws_min, nws_min, min_range + data_params['example_length'])
    frames = tf.range(min_range, max_range)
    parsed_features['signals'] = tf.map_fn(lambda sig: tf.gather(sig, frames, axis=0), parsed_features['signals'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def prepare_examples_for_classification(parsed_features, data_params):
    data = (parsed_features['signals'][0], parsed_features['signals'][1])
    label = tf.one_hot(tf.cast(parsed_features['delay'][0] - parsed_features['delay'][1] >= 0, tf.int32), 1)  # Binary
    # label = tf.one_hot(label, 2 * data_params['max_delay'])
    example = data, label
    return example


def prepare_examples_for_regression(parsed_features, data_params):
    data = (parsed_features['signals'][0], parsed_features['signals'][1])
    label = tf.expand_dims(tf.divide(parsed_features['delay'][0] - parsed_features['delay'][1], data_params['max_delay']), axis=-1)
    example = data, label
    return example


def prepare_examples_for_dctw(parsed_features):
    data = (parsed_features['signals'][0], parsed_features['signals'][1])
    labels = tf.zeros_like(parsed_features['signals'][0])
    example = data, labels
    return example


def prepare_examples_for_v1(parsed_features):
    data = parsed_features['signals'][0]
    labels = data
    example = data, labels
    return example


def prepare_examples_for_v2(parsed_features):
    data = parsed_features['signals'][1]
    labels = data
    example = data, labels
    return example


def base_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(parse_features_and_decode)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat))  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: limit_amount_of_samples(feat, data_params))
    return tfdataset


def dctw_pipeline(data_params):
    tfdataset = base_pipeline(data_params)
    tfdataset = tfdataset.map(prepare_examples_for_dctw)
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(4)
    return tfdataset


def v1_pipeline(data_params):
    tfdataset = base_pipeline(data_params)
    tfdataset = tfdataset.map(prepare_examples_for_v1)
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(4)
    return tfdataset


def v2_pipeline(data_params):
    tfdataset = base_pipeline(data_params)
    tfdataset = tfdataset.map(prepare_examples_for_v2)
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(4)
    return tfdataset


def softmax_pipeline(data_params):
    tfdataset = base_pipeline(data_params)
    tfdataset = tfdataset.map(lambda feat: prepare_examples_for_classification(feat, data_params))
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(4)
    return tfdataset


def regression_pipeline(data_params):
    tfdataset = base_pipeline(data_params)
    tfdataset = tfdataset.map(lambda feat: prepare_examples_for_regression(feat, data_params))
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(4)
    return tfdataset
