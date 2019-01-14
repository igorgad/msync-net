import tensorflow as tf


# from MSYNC.vggish import vggish


class MSYNCModel:
    def __init__(self, input_shape, model_params):
        self.input_shape = input_shape
        self.model = None
        self.model_params = model_params

    def build_conv_classifier(self, cnn, name='class'):
        for layer, units in enumerate(self.model_params.class_units):
            cnn = tf.keras.layers.BatchNormalization(name=name + '/bn%d' % layer)(cnn)
            cnn = tf.keras.layers.Conv2D(units, (3, 3), activation='elu', padding='same', name=name + '/conv%d' % layer)(cnn)
            cnn = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name + '/pool%d' % layer)(cnn)
        return cnn

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
        v1_input = tf.keras.Input(shape=self.input_shape, name='v1input')
        v2_input = tf.keras.Input(shape=self.input_shape, name='v2input')

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

        ecl_mat = EclDistanceMat()([v1_encoded, v2_encoded])
        ecl = ChannelMean()(ecl_mat)
        ecl = DiagMean()(ecl)
        ecl = tf.keras.layers.Softmax(name='ecl_output')(ecl)

        cnn = StopGradAndNormalize()(ecl_mat)
        cnn = self.build_conv_classifier(cnn)
        cnn = tf.keras.layers.Flatten()(cnn)
        cnn = tf.keras.layers.Dense(ecl.shape[-1], activation='softmax', name='cnn_output')(cnn)

        self.model = tf.keras.Model([v1_input, v2_input], [ecl, cnn])
        return self.model


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

        super(LogMel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // self.params.stft_step, self.params.num_mel_bins))


class DMRNLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DMRNLayer, self).__init__(**kwargs)

    def fusion_function(self, x, y):
        return x + (x + y) / 2.0

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        x = self.fusion_function(x, y)
        y = self.fusion_function(y, x)
        return x, y

    def build(self, input_shape):
        super(DMRNLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return tf.TensorShape(shape1), tf.TensorShape(shape2)


class EclDistanceMat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EclDistanceMat, self).__init__(**kwargs)

    def distance(self, x, y):
        return tf.sqrt(tf.maximum(tf.pow(x - y, 2), tf.keras.backend.epsilon()))

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        mat = tf.map_fn(lambda ri: self.distance(tf.expand_dims(x[:, ri, :], axis=1), y[:, :, :]), tf.range(tf.shape(x)[1]), dtype=tf.float32)
        mat = tf.transpose(mat, [1, 0, 2, 3])
        mat.set_shape([inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[1], inputs[0].shape[2]])
        return mat

    def build(self, input_shape):
        super(EclDistanceMat, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return tf.TensorShape((shape1[0], shape1[2], shape2[2], shape1[3]))


class DiagMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
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
        super(DiagMean, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))


class ChannelMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelMean, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        mean = tf.norm(inputs, axis=-1, keep_dims=True)
        tf.summary.image('avg_ecl_mat', mean)
        return mean

    def build(self, input_shape):
        super(ChannelMean, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1], input_shape[2], 1))


class StopGradAndNormalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StopGradAndNormalize, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        mat = tf.map_fn(lambda img: tf.image.per_image_standardization(img), inputs)
        mat = tf.keras.backend.stop_gradient(mat)
        return mat

    def build(self, input_shape):
        super(StopGradAndNormalize, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

