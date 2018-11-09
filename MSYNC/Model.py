
import tensorflow as tf
from MSYNC.vggish import vggish


class MSYNCModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_single_branch_model(self, name=''):
        input = tf.keras.Input(self.input_shape, name=name+'input')
        logmel = LogMel()(input)
        vggout = vggish(logmel, trainable=False, name=name)
        output = tf.keras.layers.Dense(128)(vggout)
        output = tf.keras.layers.LeakyReLU(alpha=0.3)(output)
        output = tf.keras.layers.Dense(64)(output)

        model = tf.keras.Model(input, output, name=name)
        model.load_weights('./saved_models/v1VGGish.h5', by_name=True)
        model.load_weights('./saved_models/v2VGGish.h5', by_name=True)
        return model

    def build_model(self):
        v1_model = self.build_single_branch_model('v1')
        v2_model = self.build_single_branch_model('v2')

        ecl_distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='EclDistance')([v1_model.output, v2_model.output])
        self.model = tf.keras.Model([v1_model.input, v2_model.input], ecl_distance)
        return self.model


class LogMel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(LogMel, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        with tf.device('/device:GPU:0'):
            inputs = tf.convert_to_tensor(inputs)

            stft = tf.abs(tf.contrib.signal.stft(inputs, 400, 160, pad_end=True))
            mel = tf.tensordot(stft, self.mel_matrix, 1)
            mel.set_shape(stft.shape[:-1].concatenate(self.mel_matrix.shape[-1:]))
            mel_log = tf.log(mel + 0.01)
            mel_log = tf.expand_dims(mel_log, -1)

        tf.summary.image('logmel', mel_log)
        return mel_log

    def build(self, input_shape):
        self.mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=64,
            num_spectrogram_bins=257,
            sample_rate=16000,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
            dtype=tf.float32,
            name=None
        )

        super(LogMel, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], tf.shape(self.mel_matrix)[-1], 1


def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.pow(x - y, 2), axis=1, keepdims=True), tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]
