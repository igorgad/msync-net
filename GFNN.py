
import tensorflow as tf
import numpy as np


class GFNN:
    """A GFNN implemented in the form of a RNN Cell."""

    def __init__(self, num_osc, dt, osc_params=None, use_hebbian_learning=False, heb_params=None):
        self._num_osc = num_osc
        self._dt = dt
        self._use_hebbian_learning = use_hebbian_learning
        self._z_state = []
        self._c_state = []

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
                                'mu1': -10.0,
                                'mu2': -10.0,
                                'eps': 4.0,
                                'k': 1.0}

        self._lamb = np.complex64(self._heb_params['lamb'])
        self._mu1 = np.complex64(self._heb_params['mu1'])
        self._mu2 = np.complex64(self._heb_params['mu2'])
        self._ec = np.complex64(self._heb_params['eps'])
        self._sqec = np.sqrt(self._ec)
        self._kc = np.complex64(self._heb_params['k'] + 1j * self._heb_params['k'])
        self._c_limit = np.abs(1 / self._sqec)

    def _initialize_states_with_noise(self, batch_size):
        rz = 0.01 * tf.random_normal([batch_size, self._num_osc], dtype=tf.float32)
        phiz = 0.01 * 2 * np.pi * tf.random_normal([batch_size, self._num_osc], dtype=tf.float32)
        self._z_state = tf.complex(rz, phiz)
        if self._use_hebbian_learning:
            rc = 0.01 * tf.random_normal([batch_size, self._num_osc, self._num_osc], dtype=tf.float32)
            phic = 0.01 * 2 * np.pi * tf.random_normal([batch_size, self._num_osc, self._num_osc], dtype=tf.float32)
            self._c_state = tf.complex(rc, phic)

    def _initialize_states_with_zeros(self, batch_size):
        z_init = tf.zeros([batch_size, self._num_osc], dtype=tf.complex64)
        self._z_state = z_init
        if self._use_hebbian_learning:
            self._c_state = tf.zeros([batch_size, self._num_osc, self._num_osc], dtype=tf.complex64)

    def _cdot(self, internal_stimulus, state, ti):
        z = tf.gather(internal_stimulus, ti, axis=-1)
        c = state

        c2 = tf.complex(tf.pow(tf.abs(c), 2), 0.0)
        c4 = tf.complex(tf.pow(tf.abs(c), 4), 0.0)

        def zmul(zi, zj):
            return tf.divide(zi, 1 - self._sqec * zi) * tf.divide(zj, 1 - self._sqec * tf.conj(zj)) * tf.reciprocal(
                1 - self._sqec * zj)

        fz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:, i], axis=-1), z), tf.range(self._num_osc),
                                    dtype=tf.complex64, parallel_iterations=128), [1, 0, 2])
        dcdt = c * (self._lamb + self._mu1 * c2 + self._ec * self._mu2 * c4 / (1 - self._ec * c2)) + self._kc * fz

        dcdt = tf.where(tf.greater(tf.real(dcdt), np.real(self._c_limit)), self._c_limit * tf.ones_like(dcdt), dcdt)
        dcdt = tf.where(tf.is_nan(tf.real(dcdt)), self._c_limit * tf.ones_like(dcdt), dcdt)

        return dcdt

    def _zdot(self, external_stimulus, state, ti):
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

    def run(self, ext_input):
        batch_size = tf.shape(ext_input)[0]
        input_size = tf.shape(ext_input)[1]

        self._initialize_states_with_zeros(batch_size)

        dt = tf.convert_to_tensor(self._dt)
        t = tf.range(0, tf.cast(input_size, tf.float32) * dt, self._dt)

        z_state = tf.contrib.integrate.odeint_fixed(lambda s, t: self._zdot(ext_input, s, tf.cast(t // dt, tf.int32)),
                                                    self._z_state, t, dt, method='rk4')
        z_state = tf.transpose(z_state, [1, 2, 0])
        self._z_state = z_state[:, :, -1]  # Update internal state with latest z_state

        if self._use_hebbian_learning:
            c_state = tf.contrib.integrate.odeint_fixed(lambda s, t: self._cdot(z_state, s, tf.cast(t // dt, tf.int32)), self._c_state, t, dt, method='rk4')
            c_state = tf.transpose(c_state, [1, 2, 3, 0])
            self._c_state = c_state[:, :, :, -1]  # Update internal state with latest c_state

        return z_state, self._c_state


class KerasLayer(tf.keras.layers.Layer):
    def __init__(self, num_osc, dt,
                 input_normalization=False, input_normalization_max_val=0.25,
                 osc_params=None, use_hebbian_learning=False, heb_params=None):

        super(KerasLayer, self).__init__()
        self.input_normalization = input_normalization
        self.input_normalization_max_val = input_normalization_max_val
        self.gfnn = GFNN(num_osc, dt, osc_params, use_hebbian_learning, heb_params)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        # Normalization can only be applied on non-complex inputs
        if self.input_normalization and inputs.dtype != tf.complex64:
            mean, var = tf.nn.moments(inputs, axes=[-1])
            inputs = self.input_normalization_max_val * (inputs - mean) / tf.sqrt(var)

        if inputs.dtype != tf.complex64:
            inputs = tf.complex(inputs, 0.0)

        z_state = tf.abs(self.gfnn.run(inputs)[0])
        return z_state

    def build(self, input_shape):
        super(KerasLayer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.gfnn._num_osc, input_shape[-1]
