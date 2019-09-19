
import numpy as np
import tensorflow as tf
import trainer.utils as utils


class MSYNCModel:
    def __init__(self, model_params):
        self.input_shape = (model_params.example_length, 2)
        self.model = None
        self.model_params = model_params

    def build_lstm_encoder_model(self, encoded, name=''):
        lstm_cell = tf.keras.layers.CuDNNLSTM if self.model_params.rnn_cell == 'LSTM' else tf.keras.layers.CuDNNGRU
        for layer, units in enumerate(self.model_params.encoder_units):
            encoded = tf.keras.layers.Bidirectional(lstm_cell(units, return_sequences=True), name=name + 'lstm_encoder/lstm' + str(layer))(encoded)
#             tf.keras.layers.Lambda(lambda feat: tf.summary.image('lstm_map_%d' % layer, tf.expand_dims(feat, axis=-1)), name=name + 'lstm_map_%d' % layer)(encoded)
        return encoded

    def build_cnn_encoder_model(self, encoded, name=''):
        for layer, units in enumerate(self.model_params.encoder_units):
            encoded = tf.keras.layers.Conv1D(filters=units, kernel_size=8-layer, activation='relu', padding='same', name=name + 'cnn_encoder/cnn' + str(layer))(encoded)
#             encoded = tf.keras.layers.BatchNormalization(name=name + 'cnn_encoder/bn' + str(layer))(encoded)
#             encoded = tf.keras.layers.ELU(name=name + 'cnn_encoder/elu' + str(layer))(encoded)
        return encoded

    def build_post_ecl_model(self, encoded, name='post_ecl'):
        for layer, units in enumerate(self.model_params.post_ecl_units):
            encoded = tf.keras.layers.Conv2D(filters=units, kernel_size=[5-layer, 5-layer], activation='relu', padding='same', name=name + '/conv' + str(layer))(encoded)
            if self.model_params.post_ecl_pooling:
                encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', name=name + '/pooling' + str(layer))(encoded)
            encoded = tf.keras.layers.BatchNormalization(name=name + '/bn' + str(layer))(encoded)
        return encoded

    def build_top_model(self, encoded, name=''):
        output = encoded
        for layer, units in enumerate(self.model_params.top_units[:-1]):
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation='relu'), name=name + 'fc_block%d/fc' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_block%d/bn' % layer)(output)
#             output = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'fc_block%d/elu' % layer)(output)
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.model_params.dropout), name=name + 'fc_block%d/dropout' % layer)(output)

        output = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name=name + 'fc_blockFinal/bn')(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.model_params.top_units[-1]), name=name + 'fc_blockFinal/fc')(output)
#         tf.keras.layers.Lambda(lambda feat: tf.summary.image('dense_map_%d' % layer, tf.expand_dims(feat, axis=-1)), name=name + 'dense_map_%d' % layer)(output)
        return output

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape, name='inputs')
        v1_input, v2_input = tf.keras.layers.Lambda(lambda concat: tf.unstack(concat, axis=-1), name='inputs_unstack')(inputs)

        v1_encoded = tf.keras.layers.Lambda(lambda ins: tf.expand_dims(ins, -1), name='v1time_units')(v1_input)
        v2_encoded = tf.keras.layers.Lambda(lambda ins: tf.expand_dims(ins, -1), name='v2time_units')(v2_input)

        if self.model_params.encoder_units:
            v1_encoded = self.build_lstm_encoder_model(v1_encoded, 'v1') if self.model_params.encoder_type == 'lstm' else self.build_cnn_encoder_model(v1_encoded, 'v1')
            v2_encoded = self.build_lstm_encoder_model(v2_encoded, 'v2') if self.model_params.encoder_type == 'lstm' else self.build_cnn_encoder_model(v2_encoded, 'v2')

        if self.model_params.dmrn:
            v1_encoded, v2_encoded = DMRNLayer()([v1_encoded, v2_encoded])

        if self.model_params.residual_connection:
            v1_encoded = tf.keras.layers.concatenate([v1_encoded, v1_logmel])
            v2_encoded = tf.keras.layers.concatenate([v2_encoded, v2_logmel])

        if self.model_params.top_units:
            v1_encoded = self.build_top_model(v1_encoded, 'v1')
            v2_encoded = self.build_top_model(v2_encoded, 'v2')
            
        ecl = EclDistanceMat()([v1_encoded, v2_encoded])

        if self.model_params.post_ecl_units:
            ecl = self.build_post_ecl_model(ecl)
            
        if self.model_params.ecl_end_strategy == 'diag_mean':
            ecl = tf.keras.layers.Lambda(lambda feat: tf.reduce_mean(feat, axis=-1, keep_dims=True), name='ecl_feat_mean')(ecl)
            ecl = DiagMean(invert_output=True)(ecl)

        if self.model_params.ecl_end_strategy == 'dense':
            ecl = tf.keras.layers.Flatten(name='final_ecl/flatten')(ecl)
            ecl = tf.keras.layers.Dense(1024, activation='relu', name='final_ecl/dense0')(ecl)
            ecl = tf.keras.layers.Dropout(self.model_params.dropout, name='final_ecl/dropout')(ecl)
            ecl = tf.keras.layers.Dense(self.input_shape[0] // 2 + self.input_shape[0] // 2 + 1, activation='linear', name='final_ecl/final_dense')(ecl)
        
        ecl = tf.keras.layers.Activation('softmax', name='ecl_output')(ecl)
        self.model = tf.keras.Model(inputs, ecl)
        return self.model

    def build_nw_model(self, num_examples=0):
        num_examples = num_examples if num_examples else self.model_params.num_examples
        model = self.model if self.model else self.build_model()
        inputs = tf.keras.Input(shape=(num_examples,) + self.input_shape, name='inputs')
        nw_ecl = tf.keras.layers.TimeDistributed(model, name='nw_ecl')(inputs)
        nw_ecl = tf.keras.layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1), name='nw_mean')(nw_ecl)
        nw_ecl = DiagMean(invert_output=True)(nw_ecl)
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
        tf.summary.scalar('kernel_bandwidth', self.bw[0])
        return ghs

    def build(self, input_shape):
        self.bw = self.add_weight('bandwidth', shape=[1], dtype=tf.float32, initializer=tf.keras.initializers.constant(self.bw, dtype=tf.float32), trainable=self.trainable)
        super(GaussianActivation, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


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
        return tf.maximum(tf.norm(tf.pow(x - y, 2), axis=-1, keepdims=True), tf.keras.backend.epsilon())

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        mat = tf.map_fn(lambda ri: self.distance(tf.expand_dims(x[:, ri, :], axis=1), y[:, :, :]), tf.range(tf.shape(x)[1]), dtype=tf.float32)
        mat = tf.transpose(mat, [1, 0, 2, 3])
        mat.set_shape([inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[1], 1])
        tf.summary.image('cost_mat', mat)
        return mat

    def build(self, input_shape):
        super(EclDistanceMat, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return tf.TensorShape((shape1[0], shape1[1], shape2[1], 1))


class DiagMean(tf.keras.layers.Layer):
    def __init__(self, invert_output=True, center_mean=True, **kwargs):
        self.invert_output = invert_output
        self.center_mean = center_mean
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
        mean = tf.map_fn(lambda ts: self.diag_mean(inputs, ts), tf.range(-num_time_steps // 2, 1 + num_time_steps // 2), dtype=tf.float32, parallel_iterations=32)
        mean = tf.transpose(mean)
        mean = mean - tf.reduce_mean(mean, axis=-1, keepdims=True) if self.center_mean else mean
        mean = -1 * mean if self.invert_output else mean
        mean.set_shape([inputs.shape[0], inputs.shape[1] // 2 + inputs.shape[2] // 2 + 1])
        return mean

    def build(self, input_shape):
        super(DiagMean, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))

    
@tf.custom_gradient
def _sdtw(D, gamma):
    N = D.shape[0]
    M = D.shape[1]

    # Calculate R
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

        ird_final = tf.while_loop(cond_i, body_i, [tf.constant(1), r_j], parallel_iterations=1, back_prop=False)
        r_j = ird_final[1]
        return tf.add(j, 1), r_j

    def cond_j(j, r_j):
        return tf.less_equal(j, M)

    jrd_final = tf.while_loop(cond_j, body_j, [tf.constant(1), r0], parallel_iterations=1, back_prop=False)
    R = jrd_final[1][1:-1, 1:-1]
    R.set_shape([N, M])
    R_ = tf.identity(R)

    # Calculate E
    inN = tf.concat([atleast_2d(tf.range(1, N + 1)), atleast_2d((M + 1) * tf.ones((M,), dtype=tf.int32))], axis=1)
    imM = tf.concat([atleast_2d((N + 1) * tf.ones((N,), dtype=tf.int32)), atleast_2d(tf.range(1, M + 1))], axis=1)
    rcopy = tf.concat([tf.tile(atleast_2d(tf.range(1, M + 1)), [N, 1]), atleast_2d(tf.tile(atleast_2d(tf.range(1, N + 1)), [1, M]))], axis=1)

    D = tf.scatter_nd(rcopy, tf.reshape(D, [-1]), [N + 2, M + 2])
    R = tf.scatter_nd(rcopy, tf.reshape(R, [-1]), [N + 2, M + 2]) + tf.scatter_nd(inN, tf.tile([-1e8], [inN.shape[0]]), [N + 2, M + 2]) + tf.scatter_nd(imM, tf.tile([-1e8], [imM.shape[0]]), [N + 2, M + 2]) + tf.scatter_nd([[int(N) + 1, int(M) + 1]], [R[N - 1, M - 1]], [N + 2, M + 2])
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
            return tf.greater_equal(i, tf.constant(0))

        ird_final = tf.while_loop(cond_i, body_i, [N, e_j], parallel_iterations=1, back_prop=False)
        e_j = ird_final[1]
        return tf.subtract(j, 1), e_j

    def cond_j(j, e_j):
        return tf.greater_equal(j, tf.constant(0))

    jrd_final = tf.while_loop(cond_j, body_j, [M, e0], parallel_iterations=1, back_prop=False)
    E = jrd_final[1][1:-1, 1:-1]
    E.set_shape([N, M])

    def grads(dy):
        return dy * E, None

    return R_, grads


class KSoftDTW(tf.keras.layers.Layer):
    def __init__(self, gamma=1.0, **kwargs):
        super(KSoftDTW, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, inputs, *args, **kwargs):
#         with tf.device('/cpu:0'):
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
        last_row = tf.gather(inputs, N-1, axis=1)  # * (1.0 / tf.range(1.0, int(M) + 1, dtype=tf.float32))
        last_col = tf.reverse(tf.gather(inputs, M-1, axis=2), axis=[1])  # * (1.0 / tf.range(int(N) + 1, 1.0, delta=-1.0, dtype=tf.float32))
        last_disc = tf.concat([last_row[:, N // 2:], last_col[:, :M // 2 + 1]], axis=1)
        last_disc = -1 * last_disc if self.invert_output else last_disc
        return last_disc

    def build(self, input_shape):
        super(CrossDiscrepancy, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))
