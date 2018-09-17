
# Author - Igor Pereira
# UFRN - ISASI

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def single_oscillator(alpha, omega, beta1, beta2, delta1, delta2, eps, k, dt, x_in):
    z = tf.Variable(tf.complex(0.01, 0.0))
    x = tf.complex(x_in, 0.0)

    a = tf.complex(alpha, omega)
    b = tf.complex(beta1, delta1)
    d = tf.complex(beta2, delta2)
    z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
    z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
    z_ = tf.conj(z)
    passive = x * tf.reciprocal(1.0 - tf.sqrt(eps) * x)
    active = tf.reciprocal(1.0 - tf.sqrt(eps) * z_)

    z_step = dt * (tf.multiply(z, a + b * z2 + d * eps * z4 / (1 - eps * z2)) + k * passive * active)
    step = z.assign(z + z_step)
    return step


dt = 0.01667  # Fs = 60Hz
alpha = -0.5
omega = 2.0*np.pi
beta1 = -1.0
beta2 = 0.0
delta1 = -1.0
delta2 = 0.0
eps = np.complex64(1.0 + 0j)
k = np.complex64(1.0 + 0j)


x_in = tf.placeholder(tf.float32)
osc = single_oscillator(alpha, omega, beta1, beta2, delta1, delta2, eps, k, dt, x_in)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

dt = 0.01667
t = np.arange(0, 20, dt)
sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*2*np.pi),  np.zeros(t.size//2)])

r = []
for i in sin:
    r.append(sess.run(osc, {x_in: i}))

plt.plot(sin)
plt.plot(r)
