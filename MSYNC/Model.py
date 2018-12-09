
import tensorflow as tf
# from MSYNC.vggish import vggish


class MSYNCModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.dropout_rate = 0.5

    def build_single_branch_model(self, name=''):
        input = tf.keras.Input(shape=self.input_shape, name=name+'input')       
        logmel = tf.keras.layers.TimeDistributed(LogMel(), name=name+'logmel')(input)

        # vggout = vggish(logmel, name=name)
        encoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(128, return_sequences=True), name=name+'lstm_encoder/lstm0')(logmel)
        encoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(128), name=name + 'lstm_encoder/lstm1')(encoded)
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.RepeatVector(96), name=name+'lstm_decoder/rept_vec')(encoded)
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(128, return_sequences=True), name=name+'lstm_decoder/lstm1')(decoded)
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(128, return_sequences=True), name=name + 'lstm_decoder/lstm0')(decoded)

        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), name=name + 'fc_block1/fc')(encoded)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_block1/bn')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'fc_block1/elu')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), name=name + 'fc_block2/fc')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_block2/bn')(output)

        model = tf.keras.Model(input, [decoded, output], name=name)
        return model

    def build_model(self):
        v1_model = self.build_single_branch_model('v1')
        v2_model = self.build_single_branch_model('v2')

        v1_mse = MSELayer(name='v1ae')([v1_model.output[0], v2_model.get_layer('v2logmel').output])
        v2_mse = MSELayer(name='v2ae')([v2_model.output[0], v1_model.get_layer('v1logmel').output])

        ecl_mat_distance = EclDistanceMat()([v1_model.output[1], v2_model.output[1]])
        ecl_mean_distance = DiagMean()(ecl_mat_distance)        
        ecl_softmax = tf.keras.layers.Softmax(name='ecl_softmax')(ecl_mean_distance)

        self.model = tf.keras.Model([v1_model.input, v2_model.input], [v1_mse, v2_mse, ecl_softmax])
        return self.model


class MSELayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MSELayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        yp, yt = inputs
        return tf.expand_dims(tf.reduce_mean(tf.pow(yp - yt, 2), axis=[-1, -2, -3]), axis=-1)

    def build(self, input_shape):
        super(MSELayer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1))


class LogMel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(LogMel, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        output = tf.abs(tf.contrib.signal.stft(inputs, 1600, 160, pad_end=True))
        output = tf.tensordot(output, self.mel_matrix, 1)
        output.set_shape(output.shape[:-1].concatenate(self.mel_matrix.shape[-1:]))
        output = tf.log(output + 0.01)
        output = tf.expand_dims(output, -1)
        output = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), output)
        tf.summary.image('mel_log', output)
        output = tf.squeeze(output, -1)
        return output

    def build(self, input_shape):
        self.mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=128,
            num_spectrogram_bins=1025,
            sample_rate=16000,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
            dtype=tf.float32,
            name=None
        )

        super(LogMel, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 160, 128))


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
        mat.set_shape([inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[1], 1])
        tf.summary.image('ecldist_mat', mat)
        return mat

    def build(self, input_shape):
        super(EclDistanceMat, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return tf.TensorShape((shape1[0], shape1[2], shape2[2], 1))


class DiagMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(DiagMean, self).__init__(**kwargs)

    def diag_mean(self, mat, diagi):
        ny = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1] - 1, tf.abs(tf.maximum(diagi, 0))))
        nx = tf.add(ny, diagi)
        flat_indices = ny * tf.shape(mat)[1] + nx
        flat_mat = tf.reshape(mat, [tf.shape(mat)[0], -1])
        return -1 * tf.reduce_mean(tf.gather(flat_mat, flat_indices, axis=1), axis=1)

    def call(self, inputs, *args, **kwargs):
        num_time_steps = tf.shape(inputs)[1]
        mean = tf.map_fn(lambda ts: self.diag_mean(inputs, ts), tf.range(-num_time_steps//2, 1 + num_time_steps//2), dtype=tf.float32)
        mean = tf.transpose(mean)
        mean.set_shape([inputs.shape[0], inputs.shape[1]//2 + inputs.shape[2]//2 + 1])
        return mean

    def build(self, input_shape):
        super(DiagMean, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1]//2 + input_shape[2]//2 + 1))
