
import tensorflow as tf
import numpy as np
import os
import soundfile as sf
import librosa
import io
import copy

features = {'folder': tf.VarLenFeature(tf.string), 
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
    return tf.logical_and(tf.reduce_any(tf.equal(parsed_features['instruments'], data_params.instrument_1)), tf.reduce_any(tf.equal(parsed_features['instruments'], data_params.instrument_2)))


def select_instruments(parsed_features, data_params):
    i1_index = tf.where(tf.equal(parsed_features['instruments'], data_params.instrument_1))[:, 0]
    i2_index = tf.where(tf.equal(parsed_features['instruments'], data_params.instrument_2))[:, 0]
    idx = tf.concat([i1_index, i2_index], axis=0)

    parsed_features['files'] = tf.map_fn(lambda i: tf.gather(parsed_features['files'], i, axis=0), idx, dtype=tf.string)
    parsed_features['instruments'] = tf.map_fn(lambda i: tf.gather(parsed_features['instruments'], i, axis=0), idx, dtype=tf.string)
    parsed_features['types'] = tf.map_fn(lambda i: tf.gather(parsed_features['types'], i, axis=0), idx, dtype=tf.string)
    parsed_features['activations'] = tf.map_fn(lambda i: tf.gather(parsed_features['activations'], i, axis=0), tf.concat([[0], i1_index + 1, i2_index + 1], axis=0), dtype=tf.float32)
    return parsed_features


def load_audio(parsed_features, data_params):
    def load_file(file):
        filename = os.fsencode(data_params.dataset_audio_dir) + b'/' + parsed_features['folder'] + b'/' + file
        if data_params.from_bucket:
            audio_binary = tf.py_func(lambda path: tf.gfile.Open(path, 'rb').read(), [filename], [tf.string])[0]
        else:
            audio_binary = tf.read_file(filename)

        def decode_binary(binary):
            info = sf.info(io.BytesIO(binary))
            data, sr = sf.read(io.BytesIO(binary), dtype=np.float32, frames=data_params.limit_size_seconds * info.samplerate)
            data = data.T
            data = librosa.to_mono(data)
            return librosa.resample(data, sr, data_params.sample_rate, res_type='kaiser_fast')

        smp = tf.py_func(decode_binary, [audio_binary], [tf.float32])[0]
        return smp

    parsed_features['signals'] = tf.map_fn(load_file, parsed_features['files'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def compute_activations(parsed_features, data_params):
    def func(labmat):
        dtime = np.diff(labmat[0])
        lab = []
        for b in range(1, labmat.shape[0]):
            lab.append(np.hstack([np.ones(int(dtime[i] / (1 / data_params.sample_rate)), np.float32) * labmat[b, i] for i in range(dtime.size)]))
        return np.array(lab)

    parsed_features['activations'] = tf.cond(is_empty(parsed_features['activations']), lambda: tf.ones_like(parsed_features['signals']), lambda: tf.py_func(func, [parsed_features['activations']], [tf.float32])[0])
    return parsed_features


def mix_similar_instruments(parsed_features, data_params):
    unique_instruments = tf.unique(parsed_features['instruments'])
    mix_signal = lambda signal, i: tf.reduce_mean(tf.gather(signal, tf.where(tf.equal(unique_instruments.idx, i))[:, 0], axis=0), axis=0)
    mix_types = lambda i: tf.gather(parsed_features['types'], tf.where(tf.equal(unique_instruments.idx, i))[:, 0], axis=0)[0]
    parsed_features['signals'] = tf.map_fn(lambda i: mix_signal(parsed_features['signals'], i), tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.float32, infer_shape=False)
    parsed_features['activations'] = tf.map_fn(lambda i: mix_signal(parsed_features['activations'], i), tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.float32, infer_shape=False)
    parsed_features['types'] = tf.map_fn(mix_types, tf.range(tf.shape(unique_instruments.y)[0]), dtype=tf.string, infer_shape=False)
    parsed_features['instruments'] = unique_instruments.y
    return parsed_features


def copy_v0_to_vall(parsed_features):
    num_signals = tf.shape(parsed_features['signals'])[0]
    parsed_features['signals'] = tf.map_fn(lambda sig: parsed_features['signals'][0], tf.range(num_signals), dtype=tf.float32, infer_shape=False)
    # parsed_features['instruments'] = tf.map_fn(lambda sig: parsed_features['instruments'][0], tf.range(num_signals), dtype=tf.string, infer_shape=False)
    parsed_features['activations'] = tf.map_fn(lambda sig: parsed_features['activations'][0], tf.range(num_signals), dtype=tf.float32, infer_shape=False)
    return parsed_features


def scale_signals(parsed_features, data_params):
    def scale_signal(signal):
        sig = 2 * data_params.scale_value * (signal - tf.reduce_min(signal)) / (tf.reduce_max(signal) - tf.reduce_min(signal)) - data_params.scale_value
        sig = sig - tf.reduce_mean(sig)
        return sig

    parsed_features['signals'] = tf.map_fn(scale_signal, parsed_features['signals'], dtype=tf.float32, infer_shape=False)
    return parsed_features


def generate_delay_values(parsed_features, data_params):
    delay = tf.stack([1, (data_params.max_delay - tf.random_uniform([1], 0, 2 * data_params.max_delay, dtype=tf.int32)[0])])
    parsed_features['delay'] = delay
    return parsed_features


def add_random_delay(parsed_features, data_params):
    num_signals = tf.shape(parsed_features['signals'])[0]
    delay = parsed_features['delay']

    def add_delay(signals, signal_index):
        pos_delay_func = lambda: tf.concat([tf.zeros(delay[signal_index], dtype=tf.float32), signals[signal_index, 0:-delay[signal_index]]], axis=0)
        neg_delay_func = lambda: tf.concat([signals[signal_index, tf.abs(delay[signal_index]):-1], tf.zeros(tf.abs(delay[signal_index]) + 1, dtype=tf.float32)], axis=0)

        return tf.cond(delay[signal_index] <= 0, neg_delay_func, pos_delay_func)

    parsed_features['signals'] = tf.map_fn(lambda sig: add_delay(parsed_features['signals'], sig), tf.range(num_signals), dtype=tf.float32, infer_shape=False, parallel_iterations=2)
    parsed_features['activations'] = tf.map_fn(lambda sig: add_delay(parsed_features['activations'], sig), tf.range(num_signals), dtype=tf.float32, infer_shape=False, parallel_iterations=2)
    return parsed_features


def frame_signals(parsed_features, data_params):
    parsed_features['signals'] = tf.contrib.signal.frame(parsed_features['signals'], data_params.example_length, data_params.example_length, axis=1)
    parsed_features['activations'] = tf.contrib.signal.frame(parsed_features['activations'], data_params.example_length, data_params.example_length, axis=1)
    return parsed_features


def unframe_signals(parsed_features, data_params):
    parsed_features['signals'] = tf.reshape(parsed_features['signals'], [tf.shape(parsed_features['signals'])[0], -1])
    parsed_features['activations'] = tf.reshape(parsed_features['activations'], [tf.shape(parsed_features['activations'])[0], -1])
    return parsed_features


def limit_signal_size(parsed_features, data_params):
    secs = data_params.limit_size_seconds
    parsed_features['signals'] = tf.gather(parsed_features['signals'], tf.range(tf.minimum(secs * data_params.sample_rate, tf.shape(parsed_features['signals'])[-1])), axis=-1)
    parsed_features['activations'] = tf.gather(parsed_features['activations'], tf.range(tf.minimum(secs * data_params.sample_rate, tf.shape(parsed_features['signals'])[-1])), axis=-1)
    return parsed_features


def remove_non_active_frames(parsed_features, data_params, th=0.5):
    active_frames = tf.where(tf.reduce_all(tf.greater_equal(tf.reduce_mean(parsed_features['activations'], axis=-1), th), axis=0))[:, 0]
    parsed_features['signals'] = tf.gather(parsed_features['signals'], active_frames, axis=1)
    parsed_features['activations'] = tf.gather(parsed_features['activations'], active_frames, axis=1)
    return parsed_features


def filter_nwin_less_sequential_bach(parsed_features, data_params):
    return tf.greater(tf.shape(parsed_features['signals'])[1], 1)


def random_select_frame(parsed_features, data_params):
    widx_max = tf.shape(parsed_features['signals'])[1]
    widx = tf.random_uniform([data_params.num_examples], 0, widx_max, dtype=tf.int32)

    parsed_features['signals'] = tf.gather(parsed_features['signals'], widx, axis=1)
    parsed_features['activations'] = tf.gather(parsed_features['activations'], widx, axis=1)
    parsed_features['signals'].set_shape([2, data_params.num_examples, data_params.example_length])
    return parsed_features


def compute_one_hot_delay(parsed_features, data_params):
    int_diff = tf.cast(tf.round((parsed_features['delay'][1] - parsed_features['delay'][0]) / data_params.stft_step), tf.int32)
    label_diff = int_diff + data_params.example_length // 2 // data_params.stft_step
    range_class = tf.range(label_diff - data_params.labels_precision // data_params.stft_step // 2, 1 + label_diff + data_params.labels_precision // data_params.stft_step // 2)

    one_hot_label = 1.0 * tf.one_hot(label_diff, data_params.example_length // data_params.stft_step + 1)
    one_hot_range = tf.one_hot(range_class, data_params.example_length // data_params.stft_step + 1)

    label = tf.reduce_sum(tf.concat([one_hot_range, tf.expand_dims(one_hot_label, axis=0)], axis=0), axis=0)
    label = tf.nn.softmax(label)
    parsed_features['one_hot_delay'] = label
    parsed_features['label_delay'] = label_diff
    return parsed_features


def prepare_examples(parsed_features, data_params):
    data = {'inputs': tf.stop_gradient(tf.stack([parsed_features['signals'][0], parsed_features['signals'][1]], axis=-1))}
    labels = tf.stop_gradient(parsed_features['one_hot_delay'])
    example = data, labels
    return example


def resample_train_test(parsed_features, data_params):
    parsed_features['is_train'] = tf.less_equal(tf.random_uniform([1], 0, 100, dtype=tf.float32, seed=data_params.split_seed)[0], data_params.split_rate * 100)
    return parsed_features


def select_train_examples(parsed_features):
    return parsed_features['is_train']


def select_val_examples(parsed_features):
    return tf.logical_not(parsed_features['is_train'])


def resample_folds(parsed_features, data_params):
    parsed_features['fold'] = tf.random_uniform([1], 0, data_params.num_folds, dtype=tf.int32, seed=data_params.split_seed)[0]
    return parsed_features


def select_folds(parsed_features, folds):
    return tf.reduce_any(tf.equal(folds, parsed_features['fold']))


def cached_pipeline(data_params):
    tfdataset = tf.data.TFRecordDataset(data_params.dataset_file)
    tfdataset = tfdataset.map(lambda ex: parse_features_and_decode(ex, features))
    tfdataset = tfdataset.filter(lambda feat: filter_instruments(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: select_instruments(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: load_audio(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: compute_activations(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: mix_similar_instruments(feat, data_params), num_parallel_calls=4)
    if data_params.debug_auto:
        tfdataset = tfdataset.map(lambda feat: copy_v0_to_vall(feat), num_parallel_calls=4)  # USED FOR DEBUG ONLY
    tfdataset = tfdataset.map(lambda feat: scale_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: frame_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: remove_non_active_frames(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.filter(lambda feat: filter_nwin_less_sequential_bach(feat, data_params))
    tfdataset = tfdataset.map(lambda feat: unframe_signals(feat, data_params), num_parallel_calls=4)
    # tfdataset = tfdataset.map(lambda feat: limit_signal_size(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: resample_train_test(feat, data_params), num_parallel_calls=1)  # RANDOM, Must be non-parallel for deterministic behavior
    tfdataset = tfdataset.map(lambda feat: resample_folds(feat, data_params), num_parallel_calls=1)  # RANDOM, Must be non-parallel for deterministic behavior
    tfdataset = tfdataset.cache()
    return tfdataset


def base_pipeline(data_params, tfdataset=None):
    tfdataset = tfdataset if tfdataset else cached_pipeline(data_params)
    # tfdataset = tfdataset.repeat().shuffle(64)

    tfdataset = tfdataset.map(lambda feat: generate_delay_values(feat, data_params), num_parallel_calls=1)  # RANDOM, Must be non-parallel for deterministic behavior
    tfdataset = tfdataset.map(lambda feat: add_random_delay(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: frame_signals(feat, data_params), num_parallel_calls=4)
    tfdataset = tfdataset.map(lambda feat: compute_one_hot_delay(feat, data_params), num_parallel_calls=4)
    return tfdataset


def pipeline(data_params):
    with tf.device('/cpu:0'):
        tfdataset = base_pipeline(data_params)
        tfdataset = tfdataset.map(lambda feat: random_select_frame(feat, data_params), num_parallel_calls=1)  # RANDOM, Must be non-parallel for deterministic behavior

        train_dataset = tfdataset.filter(select_train_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)
        val_dataset = tfdataset.filter(select_val_examples).map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)

        train_dataset = train_dataset.repeat().shuffle(16).batch(data_params.random_batch_size).prefetch(1)
        val_dataset = val_dataset.repeat().shuffle(16).batch(data_params.random_batch_size).prefetch(1)
    return train_dataset, val_dataset


def kfold_pipeline(data_params, train_folds, val_folds, test_folds, tfdataset=None):
    with tf.device('/cpu:0'):
        tfdataset = tfdataset if tfdataset else base_pipeline(data_params)

        train_dataset = tfdataset.filter(lambda feat: select_folds(feat, train_folds))
        val_dataset = tfdataset.filter(lambda feat: select_folds(feat, val_folds))

        train_dataset = train_dataset.map(lambda feat: random_select_frame(feat, data_params), num_parallel_calls=1)
        val_dataset = val_dataset.map(lambda feat: random_select_frame(feat, data_params), num_parallel_calls=1)

        train_dataset = train_dataset.map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)
        val_dataset = val_dataset.map(lambda feat: prepare_examples(feat, data_params), num_parallel_calls=4)

        train_dataset = train_dataset.repeat().shuffle(16).batch(data_params.random_batch_size).prefetch(1)
        val_dataset = val_dataset.repeat().shuffle(16).batch(data_params.random_batch_size).prefetch(1)

        test_dataset = tfdataset.filter(lambda feat: select_folds(feat, test_folds))
        test_params = copy.deepcopy(data_params)
        test_params.num_examples = data_params.num_examples_test
        test_params.random_batch_size = data_params.test_batch_size
        test_dataset = test_dataset.map(lambda feat: random_select_frame(feat, test_params), num_parallel_calls=1)
        test_dataset = test_dataset.map(lambda feat: prepare_examples(feat, test_params), num_parallel_calls=4)
        test_dataset = test_dataset.repeat().shuffle(16).batch(test_params.test_batch_size).prefetch(1)
    return train_dataset, val_dataset, test_dataset
