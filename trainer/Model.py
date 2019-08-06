
import numpy as np
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
        # ecl = KSoftDTW(gamma=1.0)(ecl)
        # ecl = CrossDiscrepancy()(ecl)
        # ecl = GaussianActivation(bw=6.0, trainable=True, name='gaussian_activation')(ecl)
#         ecl = tf.keras.layers.BatchNormalization()(ecl)
        ecl = DiagMean(invert_output=True)(ecl)
        ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(ecl)

        self.model = tf.keras.Model(inputs, ecl)
        return self.model

    def build_nw_model(self, num_examples=0):
        num_examples = num_examples if num_examples else self.model_params.num_examples
        model = self.model if self.model else self.build_model()
        inputs = tf.keras.Input(shape=(num_examples,) + self.input_shape, name='inputs')
        nw_ecl = tf.keras.layers.TimeDistributed(model, name='nw_ecl')(inputs)
        nw_ecl = tf.keras.layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1), name='nw_mean')(nw_ecl)
        nw_ecl = DiagMean(invert_output=False)(nw_ecl)
        nw_ecl = ProbEstimation(n=5, bw=60.0, trainable=True, name='kde_estimator')(nw_ecl)
        nw_ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(nw_ecl)

        nw_model = tf.keras.Model(inputs, nw_ecl)
        return nw_model

    def build_test_model(self, num_examples=0):
        num_examples = num_examples if num_examples else self.model_params.num_examples
        model = self.model if self.model else self.build_model()
        for layer in model.layers:
            layer.trainable = False

        inputs = tf.keras.Input(shape=(num_examples,) + self.input_shape, name='inputs')
        nw_ecl = tf.keras.layers.TimeDistributed(model, name='nw_ecl')(inputs)
        nw_ecl = tf.keras.layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1), name='nw_mean')(nw_ecl)
        nw_ecl = DiagMean(invert_output=False)(nw_ecl)
        nw_ecl = ProbEstimation(n=5, bw=60.0, trainable=True, name='kde_estimator')(nw_ecl)
        nw_ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(nw_ecl)

        self.test_model = tf.keras.Model(inputs, nw_ecl)
        return self.test_model


class ProbEstimation(tf.keras.layers.Layer):
    def __init__(self, n=5, bw=2.0, **kwargs):
        super(ProbEstimation, self).__init__(**kwargs)
        self.n = n
        self.bw = bw
        self.dist = None

    def call(self, inputs, *args, **kwargs):
        tops = utils.get_tops(inputs, self.n).indices
        flat_tops = tf.reshape(tops, [tf.shape(inputs)[0], -1])

        batch_flat_tops_mat = tf.ones((tf.shape(inputs)[0], tf.shape(flat_tops)[1], tf.shape(inputs)[-1])) * tf.expand_dims(tf.cast(flat_tops, tf.float32), axis=-1)
        batch_ntime_axis = tf.ones((tf.shape(inputs)[0], tf.shape(flat_tops)[1], tf.shape(inputs)[-1])) * tf.cast(tf.range(tf.shape(inputs)[-1]), tf.float32)

        self.dist = tf.contrib.distributions.Normal(0.0, self.bw[0])
        gaussian_mean = self.dist.prob(batch_ntime_axis - batch_flat_tops_mat)
        gaussian_mean = tf.reduce_sum(gaussian_mean, axis=1)
#         tf.summary.scalar('kde_bandwidth', self.bw[0])
        return gaussian_mean

    def build(self, input_shape):
        self.bw = self.add_weight('kde_bandwidth', shape=[1], dtype=tf.float32, initializer=tf.keras.initializers.constant(self.bw, dtype=tf.float32), trainable=self.trainable)
        super(ProbEstimation, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[-1]))


class GaussianActivation(tf.keras.layers.Layer):
    def __init__(self, bw=2.0, **kwargs):
        super(GaussianActivation, self).__init__(**kwargs)
        self.bw = bw
        self.dist = None

    def call(self, inputs, *args, **kwargs):
        dist = tf.contrib.distributions.Normal(0.0, self.bw[0])
        ghs = dist.prob(inputs)
        ghs = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), ghs)
        tf.summary.image('ecl_ghs_mat', ghs)
#         tf.summary.scalar('kernel_bandwidth', self.bw[0])
        return ghs

    def build(self, input_shape):
        reg = lambda var: 10.0 * tf.reduce_sum(tf.abs(var))
        self.bw = self.add_weight('bandwidth', shape=[1], dtype=tf.float32,
                                  initializer=tf.keras.initializers.constant(self.bw, dtype=tf.float32),
                                  regularizer=reg,
                                  trainable=self.trainable)
        super(GaussianActivation, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)



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
    def __init__(self, invert_output=True, **kwargs):
        self.mel_matrix = None
        self.invert_output = invert_output
        super(DiagMean, self).__init__(**kwargs)

    def diag_mean(self, mat, diagi):
        ny = tf.range(start=tf.abs(tf.minimum(0, diagi)), limit=tf.subtract(tf.shape(mat)[1] - 1, tf.abs(tf.maximum(diagi, 0))))
        nx = tf.add(ny, diagi)
        flat_indices = ny * tf.shape(mat)[1] + nx
        flat_mat = tf.reshape(mat, [tf.shape(mat)[0], -1])
        mean = tf.reduce_mean(tf.gather(flat_mat, flat_indices, axis=1), axis=1)
        return mean

    def call(self, inputs, *args, **kwargs):
        num_time_steps = tf.shape(inputs)[1]
        mean = tf.map_fn(lambda ts: self.diag_mean(inputs, ts), tf.range(-num_time_steps // 2, 1 + num_time_steps // 2), dtype=tf.float32)
        mean = tf.transpose(mean)
        centered_mean = mean - tf.reduce_mean(mean, axis=-1, keepdims=True)
        centered_mean = -1 * centered_mean if self.invert_output else centered_mean
        centered_mean.set_shape([inputs.shape[0], inputs.shape[1] // 2 + inputs.shape[2] // 2 + 1])
        return centered_mean

    def build(self, input_shape):
        super(DiagMean, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))


@tf.custom_gradient
def _sdtw(D, gamma):
    N = D.shape[0]
    M = D.shape[1]

    atleast_2d = lambda x: tf.reshape(x, [-1, 1])
    in0 = tf.concat([atleast_2d(tf.range(1, N + 1)), atleast_2d(tf.zeros((M,), dtype=tf.int32))], axis=1)
    im0 = tf.concat([atleast_2d(tf.zeros((N,), dtype=tf.int32)), atleast_2d(tf.range(1, M + 1))], axis=1)
    r0 = tf.scatter_nd(in0, tf.tile([1e8], [tf.shape(in0)[0]]), [N + 2, M + 2]) + tf.scatter_nd(im0, tf.tile([1e8], [tf.shape(im0)[0]]), [N + 2, M + 2])

    def body_j(j, r_j):
        def body_i(i, r_i):
            r0 = -r_i[i - 1, j - 1] / gamma
            r1 = -r_i[i - 1, j] / gamma
            r2 = -r_i[i, j - 1] / gamma
            rmax = tf.reduce_max([r0, r1, r2])
            rsum = tf.exp(r0 - rmax) + tf.exp(r1 - rmax) + tf.exp(r2 - rmax)
            softmin = - gamma * (tf.log(rsum) + rmax)
            update = tf.scatter_nd([[i, j]], [D[i - 1, j - 1] + softmin], [N + 2, M + 2])
            r_i = tf.where(tf.equal(update, 0.0), r_i, update)
            return tf.add(i, 1), r_i

        def cond_i(i, r_i):
            return tf.less_equal(i, N)

        ird_final = tf.while_loop(cond_i, body_i, [1, r_j], parallel_iterations=1)
        r_j = ird_final[1]
        return tf.add(j, 1), r_j

    def cond_j(j, r_j):
        return tf.less_equal(j, M)

    jrd_final = tf.while_loop(cond_j, body_j, [1, r0], parallel_iterations=1)
    C = jrd_final[1][1:-1, 1:-1]

    def grads(dy):
        return dy * _sdtw_backward(D, C, gamma)[1:-1, 1:-1], None
    return C, grads


def _sdtw_backward(D, R, gamma):
    N = D.shape[0]
    M = D.shape[1]

    atleast_2d = lambda x: tf.reshape(x, [-1, 1])
    inN = tf.concat([atleast_2d(tf.range(1, N + 1)), atleast_2d((M + 1) * tf.ones((M,), dtype=tf.int32))], axis=1)
    imM = tf.concat([atleast_2d((N + 1) * tf.ones((N,), dtype=tf.int32)), atleast_2d(tf.range(1, M + 1))], axis=1)
    rcopy = tf.concat([tf.tile(atleast_2d(tf.range(1, M + 1)), [N, 1]), atleast_2d(tf.tile(atleast_2d(tf.range(1, N + 1)), [1, M]))], axis=1)

    D = tf.scatter_nd(rcopy, tf.reshape(D, [-1]), [N + 2, M + 2])
    R = tf.scatter_nd(rcopy, tf.reshape(R, [-1]), [N + 2, M + 2]) + \
        tf.scatter_nd(inN, tf.tile([-1e8], [inN.shape[0]]), [N + 2, M + 2]) + \
        tf.scatter_nd(imM, tf.tile([-1e8], [imM.shape[0]]), [N + 2, M + 2]) + \
        tf.scatter_nd([[int(N) + 1, int(M) + 1]], [R[N - 1, M - 1]], [N + 2, M + 2])
    e0 = tf.scatter_nd([[int(N) + 1, int(M) + 1]], [1.0], [N + 2, M + 2])

    def body_j(j, e_j):
        def body_i(i, e_i):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = tf.exp(a0)
            b = tf.exp(b0)
            c = tf.exp(c0)
            update = tf.scatter_nd([[i, j]], [e_i[i + 1, j] * a + e_i[i, j + 1] * b + e_i[i + 1, j + 1] * c], [N + 2, M + 2])
            e_i = tf.where(tf.equal(update, 0.0), e_i, update)
            return tf.subtract(i, 1), e_i

        def cond_i(i, e_i):
            return tf.greater_equal(i, 0)

        ird_final = tf.while_loop(cond_i, body_i, [N, e_j], parallel_iterations=1)
        e_j = ird_final[1]
        return tf.subtract(j, 1), e_j

    def cond_j(j, e_j):
        return tf.greater_equal(j, 0)

    jrd_final = tf.while_loop(cond_j, body_j, [M, e0], parallel_iterations=1)
    G = jrd_final[1]
    return G


class KSoftDTW(tf.keras.layers.Layer):
    def __init__(self, gamma=1.0, **kwargs):
        super(KSoftDTW, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, inputs, *args, **kwargs):
        with tf.device('/cpu:0'):
            inputs = tf.squeeze(inputs, axis=-1)
            output = tf.map_fn(lambda bmat: _sdtw(bmat, self.gamma), inputs)
            output = tf.expand_dims(output, axis=-1)
            tf.summary.image('dtw_mat', output)
        return output

    def build(self, input_shape):
        super(KSoftDTW, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class CrossDiscrepancy(tf.keras.layers.Layer):
    def __init__(self, invert_output=True, **kwargs):
        super(CrossDiscrepancy, self).__init__(**kwargs)
        self.invert_output = invert_output

    def call(self, inputs, *args, **kwargs):
        inputs = tf.squeeze(inputs, axis=-1)
        N = inputs.shape[1]
        M = inputs.shape[2]
        last_row = tf.gather(inputs, N-1, axis=1)
        last_col = tf.reverse(tf.gather(inputs, M-1, axis=2), axis=[1])
        last_disc = tf.concat([last_row[:, N // 2:], last_col[:, :M // 2 + 1]], axis=1)
        last_disc = -1 * last_disc if self.invert_output else last_disc
        return last_disc

    def build(self, input_shape):
        super(CrossDiscrepancy, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))
