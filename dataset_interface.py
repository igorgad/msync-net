
import tensorflow as tf
import numpy as np
import os


features = {
        'folder': tf.VarLenFeature(tf.string),
        'is_train': tf.FixedLenFeature(1, tf.int64),
        'files': tf.VarLenFeature(tf.string),
        'instruments': tf.VarLenFeature(tf.string),
        'types': tf.VarLenFeature(tf.string),
        'activations': tf.VarLenFeature(tf.string)
}


def is_empty(tensor):
    return tf.equal(tf.size(tensor), 0)


def parse_features_and_decode(tf_example, features):
    parsed_features = tf.parse_single_example(tf_example, features)
    parsed_features['folder'] = tf.sparse_tensor_to_dense(parsed_features['folder'], b'')[0]
    parsed_features['is_train'] = tf.cast(parsed_features['is_train'][0], tf.bool)
    parsed_features['files'] = tf.sparse_tensor_to_dense(parsed_features['files'], b'')
    parsed_features['instruments'] = tf.sparse_tensor_to_dense(parsed_features['instruments'], b'')
    parsed_features['types'] = tf.sparse_tensor_to_dense(parsed_features['types'], b'')
    parsed_features['activations'] = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['activations'], b''), tf.float32)
    return parsed_features


def filter_instruments(parsed_features, data_params):
    return tf.logical_and(tf.reduce_any(tf.equal(parsed_features['instruments'], data_params['instrument_1'])), tf.reduce_any(tf.equal(parsed_features['instruments'], data_params['instrument_2'])))


def select_instruments(parsed_features, data_params):
    i1_index = tf.where(tf.equal(parsed_features['instruments'], data_params['instrument_1']))[:, 0]
    i2_index = tf.where(tf.equal(parsed_features['instruments'], data_params['instrument_2']))[:, 0]
    idx = tf.concat([i1_index, i2_index], axis=0)

    parsed_features['files'] = tf.map_fn(lambda i: tf.gather(parsed_features['files'], i, axis=0), idx, dtype=tf.string)
    parsed_features['instruments'] = tf.map_fn(lambda i: tf.gather(parsed_features['instruments'], i, axis=0), idx, dtype=tf.string)

    is_bach10 = is_empty(parsed_features['types'])
    parsed_features['types'] = tf.cond(is_bach10, lambda: parsed_features['types'],
                                       lambda: tf.map_fn(lambda i: tf.gather(parsed_features['types'], i, axis=0), idx, dtype=tf.string))
    parsed_features['activations'] = tf.cond(is_bach10, lambda: parsed_features['activations'],
                                             lambda: tf.map_fn(lambda i: tf.gather(parsed_features['activations'], i, axis=0), tf.concat([[0], i1_index + 1, i2_index + 1], axis=0), dtype=tf.float32))
    return parsed_features


def load_audio(parsed_features, data_params):
    def load_file(file):
        audio_binary = tf.read_file(os.fsencode(data_params['audio_root']) + b'/' + parsed_features['folder'] + b'/' + file)
        smp = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=data_params['sample_rate'], channel_count=1)
        return smp[:, 0]

    parsed_features['signals'] = tf.map_fn(load_file, parsed_features['files'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def compute_activations(parsed_features, data_params):
    def func(labmat):
        dtime = np.diff(labmat[0])
        lab = []
        for b in range(1, labmat.shape[0]):
            lab.append(np.hstack([np.ones(int(dtime[i] / (1 / data_params['sample_rate'])), np.float32) * labmat[b, i] for i in range(dtime.size)]))
        return np.array(lab)

    parsed_features['activations'] = tf.py_func(func, [parsed_features['activations']], [tf.float32])[0]
    return parsed_features


def mix_similar_instruments(parsed_features, data_params):
    unique_instruments = tf.unique(parsed_features['instruments'])
    mix_signal = lambda signal, i: tf.reduce_mean(tf.gather(signal, tf.where(tf.equal(unique_instruments.idx, i))[:, 0], axis=0), axis=0)
    mix_types = lambda i: tf.gather(parsed_features['types'], tf.where(tf.equal(unique_instruments.idx, i))[:, 0], axis=0)[0]
    parsed_features['signals'] = tf.map_fn(lambda i: mix_signal(parsed_features['signals'], i), tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.float32, infer_shape=False)
    parsed_features['activations'] = tf.map_fn(lambda i: mix_signal(parsed_features['activations'], i), tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.float32, infer_shape=False)
    parsed_features['types'] = tf.map_fn(mix_types, tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.string, infer_shape=False)
    parsed_features['instruments'] = unique_instruments.y
    return  parsed_features


def copy_v0_to_vall(parsed_features):
    num_signals = tf.shape(parsed_features['signals'])[0]
    parsed_features['signals'] = tf.map_fn(lambda sig: parsed_features['signals'][0], tf.range(num_signals), dtype=tf.float32, infer_shape=False)
    # parsed_features['instruments'] = tf.map_fn(lambda sig: parsed_features['instruments'][0], tf.range(num_signals), dtype=tf.string, infer_shape=False)
    is_bach10 = is_empty(parsed_features['types'])
    parsed_features['activations'] = tf.cond(is_bach10, lambda: parsed_features['activations'],
                                                        lambda: tf.map_fn(lambda sig: parsed_features['activations'][0], tf.range(num_signals), dtype=tf.float32, infer_shape=False))
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


def frame_signals(parsed_features, data_params):
    def frame_signal(signal):
        return tf.contrib.signal.frame(signal, data_params['example_length'], data_params['example_length'])

    parsed_features['signals'] = tf.map_fn(frame_signal, parsed_features['signals'], dtype=tf.float32)

    is_bach10 = is_empty(parsed_features['types'])
    parsed_features['activations'] = tf.cond(is_bach10, lambda: parsed_features['activations'],
                                                        lambda: tf.map_fn(frame_signal, parsed_features['activations'], dtype=tf.float32))
    return parsed_features


def remove_non_active_frames(parsed_features, data_params):
    active_frames = tf.where(tf.reduce_all(tf.greater_equal(tf.reduce_mean(parsed_features['activations'], axis=-1), 0.5), axis=0))[:, 0]
    parsed_features['signals'] = tf.squeeze(tf.map_fn(lambda sig: tf.gather(sig, active_frames, axis=0), parsed_features['signals'], dtype=tf.float32))
    parsed_features['activations'] = tf.squeeze(tf.map_fn(lambda sig: tf.gather(sig, active_frames, axis=0), parsed_features['activations'], dtype=tf.float32))
    return parsed_features


def filter_nwin_less_sequential_bach(parsed_features, data_params):
    return tf.greater(tf.shape(parsed_features['signals'])[1], data_params['sequential_batch_size'])


def random_batch(parsed_features, data_params):
    widx_max = tf.reduce_min(tf.map_fn(lambda sig: tf.shape(sig)[0], parsed_features['signals'], dtype=tf.int32, infer_shape=False))
    widx = tf.random_uniform([1], 0, widx_max, dtype=tf.int32)[0]
    parsed_features['signals'] = tf.map_fn(lambda sig: tf.gather(sig, widx, axis=0), parsed_features['signals'], dtype=tf.float32, infer_shape=False)
    parsed_features['signals'].set_shape([2, data_params['example_length']])
    return parsed_features


def sequential_batch(parsed_features, data_params):
    num_sig = tf.shape(parsed_features['signals'])[0]
    widx_max = tf.reduce_min(tf.map_fn(lambda sig: tf.shape(sig)[0], parsed_features['signals'], dtype=tf.int32, infer_shape=False))
    widx_beg = tf.random_uniform([1], 0, widx_max - data_params['sequential_batch_size'], dtype=tf.int32)[0]
    widx = tf.range(widx_beg, widx_beg + data_params['sequential_batch_size'])
    widx_mid = tf.ones(data_params['sequential_batch_size'], dtype=tf.int32) * ((widx[0] + widx[-1]) // 2 + 1)
    parsed_features['signals'] = tf.map_fn(lambda sigid: tf.where(tf.equal(sigid, 0),
                                                                  tf.gather(parsed_features['signals'][sigid], widx_mid, axis=0),
                                                                  tf.gather(parsed_features['signals'][sigid], widx, axis=0)),
                                        tf.range(num_sig), dtype=tf.float32)

    parsed_features['apply_delay'] = tf.one_hot([widx_mid[0] - widx[0]], data_params['sequential_batch_size'], on_value=False, off_value=True, dtype=tf.bool)[0]
    parsed_features['signals'].set_shape([2, data_params['sequential_batch_size'], data_params['example_length']])
    return parsed_features


def prepare_examples(parsed_features, data_params):
    data = {'v1input': parsed_features['signals'][0], 'v2input': parsed_features['signals'][1]}
    labels = tf.cast(tf.logical_not(parsed_features['apply_delay']), tf.int32)
    example = data, labels
    return example


def resample_train_test(parsed_features, train_test_ratio=0.8):
    parsed_features['is_train'] = tf.less_equal(tf.random_uniform([1], 0, 100, dtype=tf.float32)[0], train_test_ratio * 100)
    return parsed_features


def select_train_examples(parsed_features):
    return parsed_features['is_train']


def select_val_examples(parsed_features):
    return tf.logical_not(parsed_features['is_train'])


def bach10_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(lambda ex: parse_features_and_decode(ex, features))
    tfdataset = tfdataset.filter(lambda feat: filter_instruments(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: select_instruments(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params), num_parallel_calls=4)
    if data_params['debug_auto']:
        tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params), num_parallel_calls=4).cache()
    tfdataset = tfdataset.map(lambda feat: select_example_to_apply_delay(feat), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: frame_signals(feat, data_params), num_parallel_calls=4)
    
    train_dataset = tfdataset.map(lambda feat: random_batch(feat, data_params), num_parallel_calls=4)
    val_dataset = tfdataset.map(lambda feat: sequential_batch(feat, data_params), num_parallel_calls=4)

    train_dataset = train_dataset.filter(select_train_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)
    val_dataset = val_dataset.filter(select_val_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)

    train_dataset = train_dataset.repeat().shuffle(data_params['shuffle_buffer']).batch(data_params['random_batch_size'])
    val_dataset = val_dataset.repeat().shuffle(data_params['shuffle_buffer'])
    return train_dataset, val_dataset


def medleydb_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params['dataset_file'])
    tfdataset = tfdataset.map(lambda ex: parse_features_and_decode(ex, features))
    tfdataset = tfdataset.filter(lambda feat: filter_instruments(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: select_instruments(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: compute_activations(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: mix_similar_instruments(feat, data_params), num_parallel_calls=4)
    if data_params['debug_auto']:
        tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params), num_parallel_calls=4).cache()
    tfdataset = tfdataset.map(lambda feat: select_example_to_apply_delay(feat), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: frame_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: remove_non_active_frames(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.filter(lambda feat: filter_nwin_less_sequential_bach(feat, data_params))
    
    train_dataset = tfdataset.map(lambda feat: random_batch(feat, data_params), num_parallel_calls=4)
    val_dataset = tfdataset.map(lambda feat: sequential_batch(feat, data_params), num_parallel_calls=4)

    train_dataset = train_dataset.filter(select_train_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)
    val_dataset = val_dataset.filter(select_val_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)

    train_dataset = train_dataset.repeat().shuffle(data_params['shuffle_buffer']).batch(data_params['random_batch_size'])
    val_dataset = val_dataset.repeat().shuffle(data_params['shuffle_buffer'])
    return train_dataset, val_dataset
