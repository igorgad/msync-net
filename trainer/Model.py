import tensorflow as tf
import trainer.utils as utils


class MSYNCModel:
    def __init__(self, model_params):
        self.input_shape = (model_params.example_length, 2)
        self.model = None
        self.model_params = model_params

    def build_lstm_encoder_model(self, encoded, name=''):
        lstm_cell = tf.keras.layers.CuDNNLSTM if self.model_params.culstm else tf.keras.layers.LSTM
        for layer, units in enumerate(self.model_params.encoder_units):
            encoded = tf.keras.layers.Bidirectional(lstm_cell(units, return_sequences=True), name=name + 'lstm_encoder/lstm' + str(layer))(encoded)
        return encoded

    def build_top_model(self, encoded, name=''):
        output = encoded
        for layer, units in enumerate(self.model_params.top_units[:-1]):
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units), name=name + 'fc_block%d/fc' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_block%d/bn' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'fc_block%d/elu' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.model_params.dropout), name=name + 'fc_block%d/dropout' % layer)(output)

        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.model_params.top_units[-1]), name=name + 'fc_blockFinal/fc')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_blockFinal/bn')(output)
        return output

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape, name='inputs')
        v1_input, v2_input = tf.keras.layers.Lambda(lambda concat: tf.unstack(concat, axis=-1), name='inputs_unstack')(inputs)

        v1_logmel = LogMel(params=self.model_params, name='v1logmel')(v1_input)
        v2_logmel = LogMel(params=self.model_params, name='v2logmel')(v2_input)

        v1_encoded = self.build_lstm_encoder_model(v1_logmel, 'v1')
        v2_encoded = self.build_lstm_encoder_model(v2_logmel, 'v2')

        if self.model_params.dmrn:
            v1_encoded, v2_encoded = DMRNLayer()([v1_encoded, v2_encoded])

        if self.model_params.residual_connection:
            v1_encoded = tf.keras.layers.concatenate([v1_encoded, v1_logmel])
            v2_encoded = tf.keras.layers.concatenate([v2_encoded, v2_logmel])

        if self.model_params.top_units:
            v1_encoded = self.build_top_model(v1_encoded, 'v1')
            v2_encoded = self.build_top_model(v2_encoded, 'v2')

        ecl = EclDistanceMat()([v1_encoded, v2_encoded])
        #         ecl = DiagMean()(ecl)
        #         ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(ecl)

        self.model = tf.keras.Model(inputs, ecl)
        return self.model

    def build_nw_model(self, num_examples=0):
        num_examples = num_examples if num_examples else self.model_params.num_examples
        model = self.model if self.model else self.build_model()
        inputs = tf.keras.Input(shape=(num_examples,) + self.input_shape, name='inputs')
        nw_ecl = tf.keras.layers.TimeDistributed(model, name='nw_ecl')(inputs)
        nw_ecl = tf.keras.layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1), name='nw_mean')(nw_ecl)
        nw_ecl = DiagMean()(nw_ecl)
        nw_ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(nw_ecl)

        nw_model = tf.keras.Model(inputs, nw_ecl)
        return nw_model

    def build_test_model(self, num_examples=0):
        num_examples = num_examples if num_examples else self.model_params.num_examples
        model = self.model if self.model else self.build_model()

        inputs = tf.keras.Input(shape=(num_examples,) + self.input_shape, name='inputs')
        nw_ecl = tf.keras.layers.TimeDistributed(model, name='nw_ecl')(inputs)
        nw_ecl = tf.keras.layers.TimeDistributed(DiagMean(), name='diag_mean')(nw_ecl)
        nw_ecl = ProbEstimation(n=5, bw=10.0)(nw_ecl)
        nw_ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(nw_ecl)

        self.test_model = tf.keras.Model(inputs, nw_ecl)
        return self.test_model


class ProbEstimation(tf.keras.layers.Layer):
    def __init__(self, n=5, bw=2.0, **kwargs):
        self.n = n
        self.bw = bw
        self.dist = None
        super(ProbEstimation, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        tops = tf.map_fn(lambda t: utils.get_tops(tf.gather(inputs, t, axis=1), n=self.n).indices, tf.range(tf.shape(inputs)[1]), dtype=tf.int32)
        flat_tops = tf.reshape(tops, [tf.shape(inputs)[0], -1])

        batch_flat_tops_mat = tf.ones((tf.shape(inputs)[0], tf.shape(flat_tops)[1], tf.shape(inputs)[-1])) * tf.expand_dims(tf.cast(flat_tops, tf.float32), axis=-1)
        batch_ntime_axis = tf.ones((tf.shape(inputs)[0], tf.shape(flat_tops)[1], tf.shape(inputs)[-1])) * tf.cast(tf.range(tf.shape(inputs)[-1]), tf.float32)

        gaussian_mean = self.dist.prob(batch_ntime_axis - batch_flat_tops_mat)
        gaussian_mean = tf.reduce_sum(gaussian_mean, axis=1)
        return gaussian_mean

    def build(self, input_shape):
        self.dist = tf.contrib.distributions.Normal(0.0, self.bw)
        super(ProbEstimation, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[-1]))


class LogMel(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        self.mel_matrix = None
        self.params = params
        super(LogMel, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        output = tf.abs(tf.contrib.signal.stft(inputs, self.params.stft_window, self.params.stft_step, pad_end=True))
        output = tf.tensordot(output, self.mel_matrix, 1)
        output.set_shape(output.shape[:-1].concatenate(self.mel_matrix.shape[-1:]))
        output = tf.log(output + 0.01)
        output = tf.expand_dims(output, -1)
        output = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), output)
        tf.summary.image('mel_log', output)
        output = tf.squeeze(output, -1)
        return output

    def build(self, input_shape):
        self.mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins=self.params.num_mel_bins, num_spectrogram_bins=self.params.num_spectrogram_bins, sample_rate=self.params.sample_rate, lower_edge_hertz=self.params.lower_edge_hertz, upper_edge_hertz=self.params.upper_edge_hertz,
                                                                        dtype=tf.float32, name=None)

        super(LogMel, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // self.params.stft_step, self.params.num_mel_bins))


class DMRNLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(DMRNLayer, self).__init__(**kwargs)

    def fusion_function(self, x, y):
        return x + (x + y) / 2.0

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        x = self.fusion_function(x, y)
        y = self.fusion_function(y, x)
        return x, y

    def build(self, input_shape):
        super(DMRNLayer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return tf.TensorShape(shape1), tf.TensorShape(shape2)


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
        return tf.TensorShape((shape1[0], shape1[1], shape2[1], 1))


class DiagMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.mel_matrix = None
        super(DiagMean, self).__init__(**kwargs)

    def diag_mean(self, mat, diagi):
        ny = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1] - 1, tf.abs(tf.maximum(diagi, 0))))
        nx = tf.add(ny, diagi)
        flat_indices = ny * tf.shape(mat)[1] + nx
        flat_mat = tf.reshape(mat, [tf.shape(mat)[0], -1])
        mean = tf.reduce_mean(tf.gather(flat_mat, flat_indices, axis=1), axis=1)
        centered_mean = mean - tf.reduce_mean(mean, axis=-1)
        return -1 * centered_mean

    def call(self, inputs, *args, **kwargs):
        num_time_steps = tf.shape(inputs)[1]
        mean = tf.map_fn(lambda ts: self.diag_mean(inputs, ts), tf.range(-num_time_steps // 2, 1 + num_time_steps // 2), dtype=tf.float32)
        mean = tf.transpose(mean)
        mean.set_shape([inputs.shape[0], inputs.shape[1] // 2 + inputs.shape[2] // 2 + 1])
        return mean

    def build(self, input_shape):
        super(DiagMean, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))
