
import tensorflow as tf
import numpy as np


class GFNN():
    """A GFNN implemented in the form of a RNN Cell."""

    def __init__(self, num_osc, dt, avoid_nan=False, osc_params=None, use_hebbian_learning=False, heb_params=None):
        self._num_osc = num_osc
        self._dt = dt
        self._use_hebbian_learning = use_hebbian_learning
        self._avoid_nan = avoid_nan

        if osc_params is not None:
            self._osc_params = osc_params
        else:
            self._osc_params = {'f_min': 0.125,
                                'f_max': 8.0,
                                'alpha': 0.0,
                                'beta1': -1.0,
                                'beta2': 0.0,
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
        self._k = np.complex64(self._osc_params['k'] + 1j * self._osc_params['k'])

        if heb_params is not None:
            self._heb_params = heb_params
        else:
            self._heb_params = {'lamb': 0.001,
                                'mu1': -1.0,
                                'mu2': -50.0,
                                'eps': 16.0,
                                'k': 1.0}

        self._lamb = np.complex64(self._heb_params['lamb'])
        self._mu1 = np.complex64(self._heb_params['mu1'])
        self._mu2 = np.complex64(self._heb_params['mu2'])
        self._ec = np.complex64(self._heb_params['eps'])
        self._sqec = np.sqrt(self._ec)
        self._kc = np.complex64(self._heb_params['k'] + 1j * self._heb_params['k'])
        self._c_limit = np.abs(1 / self._sqec)

    def noisy_zero_state(self, batch_size):
        rz = 0.01 * tf.random_normal([batch_size, self._num_osc], dtype=tf.float32)
        phiz = 0.01 * 2 * np.pi * tf.random_normal([batch_size, self._num_osc], dtype=tf.float32)
        z_init = tf.complex(rz, phiz)

        rc = 0.01 * tf.random_normal([batch_size, self._num_osc * self._num_osc], dtype=tf.float32)
        phic = 0.01 * 2 * np.pi * tf.random_normal([batch_size, self._num_osc * self._num_osc], dtype=tf.float32)
        c_init = tf.complex(rc, phic)

        return tf.concat([z_init, c_init], axis=-1)

    def zero_state(self, batch_size):
        z_init = tf.zeros([batch_size, self._num_osc], dtype=tf.complex64)
        c_init = tf.zeros([batch_size, self._num_osc * self._num_osc], dtype=tf.complex64)
        return tf.concat([z_init, c_init], axis=-1)

    def _zdot(self, input, state, ti):
        # Extract z and c states
        z = tf.gather(state, tf.range(self._num_osc), axis=-1)
        c = tf.gather(state, tf.range(self._num_osc, self._num_osc * self._num_osc + self._num_osc), axis=-1)
        c = tf.reshape(c, [-1, self._num_osc, self._num_osc])
        c_zero_diag = tf.matrix_set_diag(c, tf.zeros([tf.shape(c)[0], self._num_osc], dtype=tf.complex64))

        # Calculate input with Connections with x(t)
        x = tf.expand_dims(tf.gather(input, ti, axis=-1), axis=-1) + 0.0 * tf.squeeze(tf.matmul(c_zero_diag, tf.expand_dims(z, axis=-1)), axis=-1)

        # Oscillators Update Rule
        z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
        z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
        z_ = tf.conj(z)
        passive = tf.divide(x, 1.0 - self._sqe * x)
        active = tf.reciprocal(1.0 - self._sqe * z_)

        dzdt = z * (self._a + self._b * z2 + self._d * self._e * z4 / (1 - self._e * z2)) + self._k * passive * active * self._f
        dcdt = c

        # Connectivity Update Rule
        if self._use_hebbian_learning:
            c2 = tf.complex(tf.pow(tf.abs(c), 2), 0.0)
            c4 = tf.complex(tf.pow(tf.abs(c), 4), 0.0)

            def zmul(zi, zj):
                return tf.divide(zi, 1 - self._sqec * zi) * tf.divide(zj, 1 - self._sqec * tf.conj(zj)) * tf.reciprocal(1 - self._sqec * zj)

            fz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:, i], axis=-1), z), tf.range(self._num_osc), dtype=tf.complex64, parallel_iterations=128), [1, 0, 2])
            fz = tf.matrix_set_diag(fz, tf.zeros([tf.shape(fz)[0], self._num_osc], dtype=tf.complex64))
            dcdt = c * (self._lamb + self._mu1 * c2 + self._ec * self._mu2 * c4 / (1 - self._ec * c2)) + self._kc * fz

            if self._avoid_nan:
                # dzdt = tf.where(tf.is_nan(tf.real(dzdt)), tf.zeros_like(dzdt), dzdt)
                dcdt = tf.where(tf.is_nan(tf.real(dcdt)), tf.zeros_like(dcdt), dcdt)
                dcdt = tf.where(tf.is_inf(tf.real(dcdt)), self._c_limit * tf.ones_like(dcdt), dcdt)

        return tf.concat([dzdt, tf.reshape(dcdt, [-1, self._num_osc * self._num_osc])], axis=-1)

    def gfnn(self, input):
        batch_size = tf.shape(input)[0]
        input_size = tf.shape(input)[1]

        dt = tf.convert_to_tensor(self._dt)
        t = tf.range(0, tf.cast(input_size, tf.float32) * dt, self._dt)
        initial_state = self.zero_state(batch_size)
        
        state = tf.contrib.integrate.odeint_fixed(lambda s, t: self._zdot(input, s, tf.cast(t // dt, tf.int32)), initial_state, t, dt, method='rk4')
        state = tf.transpose(state, [1, 0, 2])
        z_state = tf.gather(state, tf.range(self._num_osc), axis=-1)
        c_state = tf.gather(state, tf.range(self._num_osc, self._num_osc * self._num_osc + self._num_osc), axis=-1)
        c_state = tf.reshape(c_state, [batch_size, -1, self._num_osc, self._num_osc])

        return z_state, c_state
