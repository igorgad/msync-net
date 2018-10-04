
import tensorflow as tf
import numpy as np


class GFNN():
    """A GFNN implemented in the form of a RNN Cell."""

    def __init__(self, num_osc, dt, scale_connections=False, osc_params=None, use_hebbian_learning=False, heb_params=None):
        self._num_osc = num_osc
        self._dt = dt
        self._use_hebbian_learning = use_hebbian_learning
        self._scale_connections = scale_connections
        self._z_state = []
        self._c_state = []

        if osc_params is not None:
            self._osc_params = osc_params
        else:
            self._osc_params = {'f_min': 20.0,
                                'f_max': 5000.0,
                                'alpha': 0.0,
                                'beta1': -1.0,
                                'beta2': 0.0,
                                'delta1': 0.0,
                                'delta2': 0.0,
                                'eps': 1.0,
                                'k': 1.0}

        self._f = np.logspace(np.log10(self._osc_params['f_min']), np.log10(self._osc_params['f_max']),
                              self._num_osc, dtype=np.float32)
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
            self._heb_params = {'lamb': 0.0,
                                'mu1': -1.0,
                                'mu2': -50.0,
                                'eps': 16.0,
                                'k': 1.0,
                                'lr': 0.01}

        self._lamb = np.complex64(self._heb_params['lamb'])
        self._mu1 = np.complex64(self._heb_params['mu1'])
        self._mu2 = np.complex64(self._heb_params['mu2'])
        self._ec = np.complex64(self._heb_params['eps'])
        self._sqec = np.sqrt(self._ec)
        self._kc = np.complex64(self._heb_params['k'] + 1j * self._heb_params['k'])
        self._c_limit = np.abs(1 / self._sqec)
        self._lr = np.complex64(self._heb_params['lr'] + 1j * self._heb_params['lr'])

    def _initialize_states_with_noise(self, batch_size):
        rz = 0.01 * tf.random_normal([batch_size, 1, self._num_osc], dtype=tf.float32)
        phiz = 0.01 * 2 * np.pi * tf.random_normal([batch_size, 1, self._num_osc], dtype=tf.float32)
        self._z_state = tf.complex(rz, phiz)
        if self._use_hebbian_learning:
            rc = 0.01 * tf.random_normal([batch_size, 1, self._num_osc, self._num_osc], dtype=tf.float32)
            phic = 0.01 * 2 * np.pi * tf.random_normal([batch_size, 1, self._num_osc, self._num_osc], dtype=tf.float32)
            self._c_state = tf.complex(rc, phic)

    def _initialize_states_with_zeros(self, batch_size):
        z_init = tf.zeros([batch_size, 1, self._num_osc], dtype=tf.complex64)
        self._z_state = z_init
        if self._use_hebbian_learning:
            self._c_state = tf.zeros([batch_size, 1, self._num_osc, self._num_osc], dtype=tf.complex64)

    def _cdot(self, internal_stimulus, state, ti):
        z = tf.gather(internal_stimulus, ti, axis=1)
        c = state

        c2 = tf.complex(tf.pow(tf.abs(c), 2), 0.0)
        c4 = tf.complex(tf.pow(tf.abs(c), 4), 0.0)

        def zmul(zi, zj):
            return tf.divide(zi, 1 - self._sqec * zi) * tf.divide(zj, 1 - self._sqec * tf.conj(zj)) * tf.reciprocal(
                1 - self._sqec * zj)

        fz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:, i], axis=-1), z), tf.range(self._num_osc),
                                    dtype=tf.complex64, parallel_iterations=128), [1, 0, 2])
        dcdt = self._lr * (c * (self._lamb + self._mu1 * c2 + self._ec * self._mu2 * c4 / (1 - self._ec * c2)) + self._kc * fz)

        if self._scale_connections:
            dcdt = tf.where(tf.greater(tf.real(dcdt), np.real(self._c_limit)), self._c_limit * tf.ones_like(dcdt), dcdt)
            dcdt = tf.where(tf.is_nan(tf.real(dcdt)), self._c_limit * tf.ones_like(dcdt), dcdt)

        return dcdt

    def _zdot(self, external_stimulus, state, ti):
        # Extract z and c states
        z = state
        x = tf.expand_dims(tf.gather(external_stimulus, ti, axis=-1), axis=-1)

        if self._use_hebbian_learning:
            x = tf.expand_dims(tf.gather(external_stimulus, ti, axis=-1), axis=-1) + tf.squeeze(
                tf.matmul(self._c_state, tf.expand_dims(z, axis=-1)), axis=-1)  # Calculate input with Connections with x(t)

        # Oscillators Update Rule
        z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
        z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
        z_ = tf.conj(z)
        passive = tf.divide(x, 1.0 - self._sqe * x)
        active = tf.reciprocal(1.0 - self._sqe * z_)

        dzdt = z * (self._a + self._b * z2 + self._d * self._e * z4 / (
                    1 - self._e * z2)) + self._k * passive * active * self._f
        return dzdt

    def gfnn(self, ext_input):
        batch_size = tf.shape(ext_input)[0]
        input_size = tf.shape(ext_input)[1]

        self._initialize_states_with_zeros(batch_size)

        dt = tf.convert_to_tensor(self._dt)
        t = tf.range(0, tf.cast(input_size, tf.float32) * dt, self._dt)

        z_state = tf.contrib.integrate.odeint_fixed(lambda s, t: self._zdot(ext_input, s, tf.cast(t // dt, tf.int32)),
                                                    self._z_state, t, dt, method='rk4')
        self._z_state = tf.transpose(tf.squeeze(z_state, axis=2), [1, 0, 2])

        if self._use_hebbian_learning:
            c_state = tf.contrib.integrate.odeint_fixed(
                lambda s, t: self._cdot(self._z_state, s, tf.cast(t // dt, tf.int32)), self._c_state, t, dt,
                method='rk4')
            self._c_state = tf.transpose(tf.squeeze(c_state, axis=2), [1, 0, 2, 3])

        return self._z_state, self._c_state
