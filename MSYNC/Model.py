
import tensorflow as tf
# from MSYNC.vggish import vggish


class MSYNCModel:
    def __init__(self, input_shape, model_params):
        self.input_shape = input_shape
        self.model = None
        self.model_params = model_params

    def build_encoder(self, encoded, name=''):
        for layer, units in enumerate(self.model_params['lstm_units'][:-1]):
            encoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(units, return_sequences=True), name=name + 'lstm_encoder/lstm' + str(layer))(encoded)
        encoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(self.model_params['lstm_units'][-1], return_sequences=False), name=name + 'lstm_encoder/lstmFinal')(encoded)
        return encoded

    def build_decoder(self, encoded, name=''):
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.RepeatVector(96), name=name + 'lstm_decoder/rept_vec')(encoded)
        for layer, units in enumerate(reversed(self.model_params['lstm_units'])):
            decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.CuDNNLSTM(units, return_sequences=True), name=name + 'lstm_decoder/lstm' + str(layer))(decoded)
        return decoded

    def build_top(self, encoded, name=''):
        output = encoded
        for layer, units in enumerate(self.model_params['top_units'][-1:]):
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), name=name + 'fc_block%d/fc' % layer)(encoded)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_block%d/bn' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'fc_block%d/elu' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.model_params['dropout']), name=name + 'fc_block%d/dropout' % layer)(output)

        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.model_params['top_units'][-1]), name=name + 'fc_blockFinal/fc')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_blockFinal/bn')(output)
        return output

    def build_model(self):
        v1_input = tf.keras.Input(shape=self.input_shape, name='v1input')
        v2_input = tf.keras.Input(shape=self.input_shape, name='v2input')

        v1_logmel = tf.keras.layers.TimeDistributed(LogMel(params=self.model_params), name='v1logmel')(v1_input)
        v2_logmel = tf.keras.layers.TimeDistributed(LogMel(params=self.model_params), name='v2logmel')(v2_input)

        v1_encoded = self.build_encoder(v1_logmel, 'v1')
        v2_encoded = self.build_encoder(v2_logmel, 'v2')
        v1_decoded = self.build_decoder(v1_encoded, 'v1')
        v2_decoded = self.build_decoder(v2_encoded, 'v2')

        v1_mse = MSELayer(name='v1ae')([v1_decoded, v1_logmel])
        v2_mse = MSELayer(name='v2ae')([v2_decoded, v2_logmel])

        v1_top = self.build_top(v1_encoded, 'v1')
        v2_top = self.build_top(v2_encoded, 'v2')

        ecl_mat_distance = EclDistanceMat()([v1_top, v2_top])
        ecl_mean_distance = DiagMean()(ecl_mat_distance)
        ecl_softmax = tf.keras.layers.Softmax(name='ecl_softmax')(ecl_mean_distance)

        self.model = tf.keras.Model([v1_input, v2_input], [v1_mse, v2_mse, ecl_softmax])
        return self.model


class MSELayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MSELayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        yp, yt = inputs
        tf.summary.image('decoded', tf.expand_dims(tf.gather(yp, 0, axis=1), axis=-1))
        return tf.expand_dims(tf.reduce_mean(tf.pow(yp - yt, 2), axis=[-1, -2, -3]), axis=-1)

    def build(self, input_shape):
        super(MSELayer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1))


class LogMel(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        self.mel_matrix = None
        self.params = params
        super(LogMel, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        output = tf.abs(tf.contrib.signal.stft(inputs, self.params['stft_window'], self.params['stft_step'], pad_end=True))
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
            num_mel_bins=self.params['num_mel_bins'],
            num_spectrogram_bins=self.params['num_spectrogram_bins'],
            sample_rate=self.params['sample_rate'],
            lower_edge_hertz=self.params['lower_edge_hertz'],
            upper_edge_hertz=self.params['upper_edge_hertz'],
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
        return tf.sqrt(tf.maximum(tf.norm(tf.pow(x - y, 2), axis=-1, keepdims=True), tf.keras.backend.epsilon()))

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
