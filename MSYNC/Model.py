
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

        ecl_distance = EclDistance()([v1_model.output, v2_model.output])
        self.model = tf.keras.Model([v1_model.input, v2_model.input], ecl_distance)
        return self.model


class LogMel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(LogMel, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
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


class EclDistanceMat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(EclDistanceMat, self).__init__(**kwargs)

    def distance(self, x, y):
        return tf.sqrt(tf.maximum(tf.reduce_sum(tf.pow(x - y, 2), axis=-1, keepdims=True), tf.keras.backend.epsilon()))

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        mat = tf.map_fn(lambda ri: self.distance(tf.expand_dims(x[:, ri, :], axis=1), y[:, :, :]), tf.range(tf.shape(x)[1]), dtype=tf.float32)
        mat = tf.transpose(mat, [1, 0, 2, 3])
        tf.summary.image('ecldist_mat', mat)
        return mat

    def build(self, input_shape):
        super(EclDistanceMat, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return [shape1[0], shape1[1], shape2[1], 1]


class DiagMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(DiagMean, self).__init__(**kwargs)

    def diag_mean(self, mat, diagi):
        nx = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1] - 1, tf.abs(diagi)))
        ny = tf.add(nx, diagi)
        indices = [tf.concat([tf.expand_dims(nx, -1), tf.expand_dims(ny, -1)], axis=1)]
        return tf.reduce_mean(tf.gather_nd(mat, indices))

    def call(self, inputs, *args, **kwargs):
        num_time_steps = tf.shape(inputs)[1]
        mean = tf.map_fn(lambda ts: self.diag_mean(inputs, ts), tf.range(-num_time_steps, num_time_steps), dtype=tf.float32)
        return mean

    def build(self, input_shape):
        super(DiagMean, self).build(input_shape)  # Be sure to call this at the end

    # def compute_output_shape(self, input_shape):
    #     shape1, shape2 = input_shape
    #     return [shape1[0], shape1[1], shape2[1], 1]