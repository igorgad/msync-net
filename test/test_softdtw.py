
import tensorflow as tf
import numpy as np
from trainer.Model import EclDistanceMat, DiagMean
import trainer.stats as stats
import trainer.dataset_interface
import matplotlib.pyplot as plt
import time


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
        last_row = tf.gather(inputs, N-1, axis=1)  # * (1.0 / tf.range(1.0, int(M) + 1, dtype=tf.float32))
        last_col = tf.reverse(tf.gather(inputs, M-1, axis=2), axis=[1])  # * (1.0 / tf.range(int(N) + 1, 1.0, delta=-1.0, dtype=tf.float32))
        last_disc = tf.concat([last_row[:, N // 2:], last_col[:, :M // 2 + 1]], axis=1)
        last_disc = -1 * last_disc if self.invert_output else last_disc
        return last_disc

    def build(self, input_shape):
        super(CrossDiscrepancy, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1] // 2 + input_shape[2] // 2 + 1))


ts = 32
siga = tf.random.normal((512, ts, 128))
sigb = tf.random.normal((512, ts, 128))
lb = tf.random.normal((512, ts + 1), mean=8)
dts_ex = tf.data.Dataset.from_tensor_slices({'i1': siga, 'i2': sigb})
dts_lb = tf.data.Dataset.from_tensor_slices(lb)
dts = tf.data.Dataset.zip((dts_ex, dts_lb)).repeat().batch(8)


ksiga = tf.keras.Input(shape=(ts, 128), name='i1')
ksigb = tf.keras.Input(shape=(ts, 128), name='i2')

lsa = tf.keras.layers.CuDNNLSTM(64, return_sequences=True)(ksiga)
lsb = tf.keras.layers.CuDNNLSTM(64, return_sequences=True)(ksigb)

dist_mat = EclDistanceMat()([lsa, lsb])
dist_acc = KSoftDTW(gamma=1.0)(dist_mat)
dist_disc = CrossDiscrepancy()(dist_acc)
dist_max = tf.keras.layers.Softmax(name='ecl_output')(dist_disc)

md = tf.keras.Model([ksiga, ksigb], dist_max)
md.summary()
md.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.01))

tensorboard = stats.TensorBoardAVE(log_dir='./logs/test_softdtw/run0', histogram_freq=1, batch_size=4, write_images=True, range=10)
md.fit(dts, validation_data=dts, steps_per_epoch=10, epochs=100, validation_steps=10, callbacks=[tensorboard])
