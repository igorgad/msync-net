
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

    def __init__(self, num_osc, dt, osc_params=None, use_hebbian_learning=False, heb_params=None):
        self._num_osc = num_osc
        self._dt = dt
        self._use_hebbian_learning = use_hebbian_learning

        if osc_params is not None:
            self._osc_params = osc_params
        else:
            self._osc_params = {'omega': 2 * np.pi * np.logspace(0, 1, self._num_osc, dtype=np.float32) / 10,
                                'alpha': -0.5 * np.ones(self._num_osc, np.float32),
                                'beta1': -1.0,
                                'beta2': -50.0,
                                'delta1': -50.0,
                                'delta2': -50.0,
                                'eps': 0.5,
                                'k': 0.0}

        self._a = tf.complex(self._osc_params['alpha'], self._osc_params['omega'])
        self._b = tf.complex(self._osc_params['beta1'], self._osc_params['delta1'])
        self._d = tf.complex(self._osc_params['beta2'], self._osc_params['delta2'])
        self._osc_params['eps'] = np.complex64(self._osc_params['eps'])
        self._osc_params['k'] = np.complex64(self._osc_params['k'])

        if heb_params is not None:
            self._heb_params = heb_params
        else:
            self._heb_params = {'lamb': 0.01,
                                'mu1': -1.0,
                                'mu2': -50.0,
                                'eps': 1.0,
                                'k': 1.0}

        self._heb_params['lamb'] = np.complex64(self._heb_params['lamb'])
        self._heb_params['mu1'] = np.complex64(self._heb_params['mu1'])
        self._heb_params['mu2'] = np.complex64(self._heb_params['mu2'])
        self._heb_params['eps'] = np.complex64(self._heb_params['eps'])
        self._heb_params['k'] = np.complex64(self._heb_params['k'])


    @property
    def state_size(self):
        return GFNNStateTuple(self._num_osc, self._num_osc * self._num_osc)

    @property
    def output_size(self):
        return self._num_osc

    def __call__(self, inputs, state, scope=None):
        """Gradient Frequency Neural Network cell (GFNN)."""

        with tf.variable_scope(scope or type(self).__name__):  # "GFNNCell"
            z, c = state
            c = tf.reshape(c, [-1, self._num_osc, self._num_osc])

            # Oscillators Update Rule
            ints = tf.squeeze(tf.matmul(c, tf.expand_dims(z, axis=-1)), axis=-1)
            x = tf.complex(inputs, 0.0) + self._osc_params['k'] * ints

            z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
            z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
            z_ = tf.conj(z)
            passive = x * tf.reciprocal(1.0 - tf.sqrt(self._osc_params['eps']) * x)
            active = tf.reciprocal(1.0 - tf.sqrt(self._osc_params['eps']) * z_)

            dzdt = self._dt * (z * (self._a + self._b * z2 + self._d * self._osc_params['eps'] * z4
                                    / (1 - self._osc_params['eps'] * z2)) + passive * active)
            new_z = z + dzdt
            new_c = tf.reshape(c, [-1, self._num_osc * self._num_osc])

            # Connectivity Matrix Update Rule (Hebbian learning)
            if self._use_hebbian_learning:
                c2 = tf.complex(tf.pow(tf.abs(c), 2), 0.0)
                c4 = tf.complex(tf.pow(tf.abs(c), 4), 0.0)

                def zmul(zi, zj):
                    return zi * tf.reciprocal(1 - tf.sqrt(self._heb_params['eps']) * zi) * zj \
                           * tf.reciprocal(1 - tf.sqrt(self._heb_params['eps']) * tf.conj(zj)) \
                           * tf.reciprocal(1 - tf.sqrt(self._heb_params['eps']) * zj)

                fz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:,i], axis=-1), z), tf.range(self._num_osc), dtype=tf.complex64), [1, 0, 2])
                dcdt = self._dt * (c * (self._heb_params['lamb'] + self._heb_params['mu1'] * c2 + self._heb_params['eps']
                                        * self._heb_params['mu2'] * c4 / (1 - self._heb_params['eps'] * c2)) + self._heb_params['k'] * fz)
                new_c = tf.reshape(c + dcdt, [-1, self._num_osc * self._num_osc])

            # Update State Tuple
            new_state = GFNNStateTuple(new_z, new_c)
            return new_z, new_state
