
import tensorflow as tf
import os

features = {
        'folder': tf.VarLenFeature(tf.string),
        'files': tf.VarLenFeature(tf.string),
        'instruments': tf.VarLenFeature(tf.string),
        'signals': tf.VarLenFeature(tf.string)
}


def parse_features_and_decode(tf_example):
    parsed_features = tf.parse_single_example(tf_example, features)
    parsed_features['folder'] = tf.sparse_tensor_to_dense(parsed_features['folder'], b'')[0]
    parsed_features['files'] = tf.sparse_tensor_to_dense(parsed_features['files'], b'')
    parsed_features['instruments'] = tf.sparse_tensor_to_dense(parsed_features['instruments'], b'')
    parsed_features['signals'] = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['signals'], b''), tf.float32)
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


def select_example_to_apply_delay(parsed_features, ratio=0.5):
    parsed_features['apply_delay'] = tf.random_uniform([1], 1, 100, dtype=tf.float32)[0] < ratio * 100
    return parsed_features


def add_random_delay(parsed_features, data_params):
    num_signals = tf.shape(parsed_features['signals'])[0]
    delay = tf.random_uniform([num_signals], 1, data_params['max_delay'], dtype=tf.int32)

    def add_delay(signal_index):
        return tf.concat([tf.zeros(delay[signal_index], dtype=tf.float32), parsed_features['signals'][signal_index, 0:-delay[signal_index]]], axis=0)

    parsed_features['delay'] = delay
    parsed_features['signals'] = tf.where(parsed_features['apply_delay'], tf.map_fn(add_delay, tf.range(num_signals), dtype=tf.float32, infer_shape=False), parsed_features['signals'])
    return parsed_features


def limit_amount_of_samples(parsed_features, data_params):
    nws_per_signal = tf.map_fn(lambda sig: tf.shape(sig)[0], parsed_features['signals'], dtype=tf.int32, infer_shape=False)
    nws_min = tf.reduce_min(nws_per_signal)
    min_range = tf.where(data_params['example_length'] > nws_min, 0, tf.random_uniform([1], 0, tf.abs(nws_min - data_params['example_length']), dtype=tf.int32)[0])
    max_range = tf.where(data_params['example_length'] > nws_min, nws_min, min_range + data_params['example_length'])
    frames = tf.range(min_range, max_range)
    parsed_features['signals'] = tf.map_fn(lambda sig: tf.gather(sig, frames, axis=0), parsed_features['signals'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def sequential_batch(parsed_features, data_params):
    def frame_signal(signal):
        return tf.contrib.signal.frame(signal, data_params['example_length'], data_params['example_length'])

    framed_signals = tf.map_fn(frame_signal, parsed_features['signals'], dtype=tf.float32)
    num_sig = tf.shape(framed_signals)[0]
    num_win = tf.shape(framed_signals[0])[0]
    idw_beg = tf.random_uniform([1], 0, num_win - data_params['sequential_batch_size'], dtype=tf.int32)[0]
    idw_end = idw_beg + data_params['sequential_batch_size']
    parsed_features['signals'] = tf.map_fn(lambda sigid: tf.where(tf.equal(sigid, 0),
                                                                  tf.gather(framed_signals[sigid], tf.ones(data_params['sequential_batch_size'], dtype=tf.int32) * (idw_beg + idw_end) // 2, axis=0),
                                                                  tf.gather(framed_signals[sigid], tf.range(idw_beg, idw_end), axis=0)),
                                        tf.range(num_sig), dtype=tf.float32)
    return parsed_features


def prepare_examples(parsed_features):
    data = {'v1input': parsed_features['signals'][0], 'v2input': parsed_features['signals'][1]}
    labels = tf.cast(tf.logical_not(parsed_features['apply_delay']), tf.int32)
    example = data, labels
    return example


def train_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(parse_features_and_decode)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: select_example_to_apply_delay(feat), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: limit_amount_of_samples(feat, data_params), num_parallel_calls=4)

    tfdataset = tfdataset.map(prepare_examples, num_parallel_calls=4)
    tfdataset = tfdataset.repeat(data_params['repeat']).shuffle(data_params['shuffle_buffer']).batch(data_params['batch_size']).prefetch(8)
    return tfdataset


def test_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(parse_features_and_decode)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: select_example_to_apply_delay(feat), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: sequential_batch(feat, data_params), num_parallel_calls=4)

    tfdataset = tfdataset.map(prepare_examples, num_parallel_calls=4)
    tfdataset = tfdataset.repeat(data_params['repeat'])
    return tfdataset
