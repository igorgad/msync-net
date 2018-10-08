
import tensorflow as tf
import numpy as np
import collections


_GFNNStateTuple = collections.namedtuple("GFNNStateTuple", ("z", "c")) # Oscilators state = z, Connectivity state = c


class GFNNStateTuple(_GFNNStateTuple):
    """Tuple used by GFNN Cells for `oscillators_state`, `connectivity_state`, and output state.
    Stores two elements: `(z, c)`, in that order.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (z, c) = self
        if not z.dtype == c.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class GFNNCell(tf.contrib.rnn.RNNCell):
    """A GFNN implemented in the form of a RNN Cell."""

    def __init__(self, num_osc, dt, avoid_nan=False, osc_params=None, use_hebbian_learning=False, heb_params=None):
        self._num_osc = num_osc
        self._dt = dt
        self._use_hebbian_learning = use_hebbian_learning
        self._avoid_nan = avoid_nan

        if osc_params is not None:
            self._osc_params = osc_params
        else:
            self._osc_params = {'f_min': 0.5,
                                'f_max': 8.0,
                                'alpha': -1.0,
                                'beta1': -1.0,
                                'beta2': -1.0,
                                'delta1': 0.0,
                                'delta2': 0.0,
                                'eps': 1.0,
                                'k': 1.0}

        self._f = np.logspace(np.log10(self._osc_params['f_min']), np.log10(self._osc_params['f_max']), self._num_osc, dtype=np.float32)
        self._osc_params['alpha'] = self._osc_params['alpha'] * np.ones(self._num_osc, np.float32)
        self._a = tf.complex(self._osc_params['alpha'], 2 * np.pi) * self._f
        self._b = tf.complex(self._osc_params['beta1'], self._osc_params['delta1']) * self._f
        self._d = tf.complex(self._osc_params['beta2'], self._osc_params['delta2']) * self._f
        self._e = np.complex64(self._osc_params['eps'])
        self._sqe = np.sqrt(self._e)
        self._k = np.complex64(self._osc_params['k'])

        if heb_params is not None:
            self._heb_params = heb_params
        else:
            self._heb_params = {'lamb': 0.0,
                                'mu1': -1.0,
                                'mu2': -50.0,
                                'eps': 1.0,
                                'k': 1.0}

        self._lamb = np.complex64(self._heb_params['lamb'])
        self._mu1 = np.complex64(self._heb_params['mu1'])
        self._mu2 = np.complex64(self._heb_params['mu2'])
        self._ec = np.complex64(self._heb_params['eps'])
        self._sqec = np.sqrt(self._ec)
        self._kc = np.complex64(self._heb_params['k'])
        self._c_limit = np.abs(1 / self._sqec)


    @property
    def state_size(self):
        return GFNNStateTuple(self._num_osc, self._num_osc * self._num_osc)

    @property
    def output_size(self):
        return self._num_osc

    def noisy_zero_state(self, batch_size):
        zero_state = self.zero_state(batch_size, dtype=tf.complex64)
        rz = 0.01 * tf.random_normal(zero_state.z.shape)
        phiz = 0.01 * 2 * np.pi * tf.random_normal(zero_state.z.shape)
        rc = 0.01 * tf.random_normal(zero_state.c.shape)
        phic = 0.01 * 2 * np.pi * tf.random_normal(zero_state.c.shape)
        return GFNNStateTuple(tf.complex(rz, 2 * np.pi * phiz), tf.complex(rc, phic))

    def __call__(self, inputs, state, scope=None):
        """Gradient Frequency Neural Network cell (GFNN)."""

        with tf.variable_scope(scope or type(self).__name__):  # "GFNNCell"
            z, c = state
            c = tf.reshape(c, [-1, self._num_osc, self._num_osc])
            c_zero_diag = tf.matrix_set_diag(c, tf.zeros([tf.shape(c)[0], self._num_osc], dtype=tf.complex64))

            # Oscillators Update Rule
            ints = tf.squeeze(tf.matmul(c_zero_diag, tf.expand_dims(z, axis=-1)), axis=-1)
            x = tf.complex(inputs, 0.0) + self._k * ints

            z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
            z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
            z_ = tf.conj(z)
            passive = tf.divide(x, 1.0 - self._sqe * x)
            active = tf.reciprocal(1.0 - self._sqe * z_)

            dzdt = self._dt * (z * (self._a + self._b * z2 + self._d * self._e * z4 / (1 - self._e * z2)) + passive * active * self._f)

            if self._avoid_nan:
                dzdt = tf.where(tf.is_nan(tf.real(dzdt)), tf.zeros_like(dzdt), dzdt)
                # dzdt = tf.where(tf.is_inf(tf.real(dzdt)), tf.ones_like(dzdt), dzdt)

            new_z = z + dzdt
            new_c = tf.reshape(c, [-1, self._num_osc * self._num_osc])

            # Connectivity Matrix Update Rule (Hebbian learning)
            if self._use_hebbian_learning:
                c2 = tf.complex(tf.pow(tf.abs(c), 2), 0.0)
                c4 = tf.complex(tf.pow(tf.abs(c), 4), 0.0)

                def zmul(zi, zj):
                    return tf.divide(zi, 1 - self._sqec * zi) * tf.divide(zj, 1 - self._sqec * tf.conj(zj)) * tf.reciprocal(1 - self._sqec * zj)

                fz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:,i], axis=-1), z), tf.range(self._num_osc), dtype=tf.complex64), [1, 0, 2])
                dcdt = self._dt * (c * (self._lamb + self._mu1 * c2 + self._ec * self._mu2 * c4 / (1 - self._ec * c2)) + self._kc * fz)

                if self._avoid_nan:
                    dcdt = tf.where(tf.is_nan(tf.real(dcdt)), tf.zeros_like(dcdt), dcdt)
                    dcdt = tf.where(tf.is_inf(tf.real(dcdt)), self._c_limit * tf.ones_like(dcdt), dcdt)

                new_c = tf.reshape(c + dcdt, [-1, self._num_osc * self._num_osc])


            # Update State Tuple
            new_state = GFNNStateTuple(new_z, new_c)
            return new_z, new_state
