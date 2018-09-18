# Author - Igor Pereira
# UFRN - ISASI

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#def single_oscillator(alpha, omega, beta1, beta2, delta1, delta2, eps, k, dt, x_in):
#    return step




dt = 0.01667  # Fs = 60Hz
t = np.arange(0, 20, dt)
sin = 0.25 * np.sin(t*2*np.pi)
#sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*1*np.pi),  np.zeros(t.size//2)])

omega = np.arange(0, 2.0*np.pi, np.pi/10, np.float32)
alpha = -0.5 * np.ones_like(omega, np.float32)
beta1 = -1.0
beta2 = 0.0
delta1 = -1.0
delta2 = 0.0
eps = np.complex64(1.0 + 0j)
k = np.complex64(1.0 + 0j)

z_init_r = 0.01 * np.ones_like(alpha, np.float32)
z_init_i = 0.00 * np.ones_like(alpha, np.float32)


x_in = tf.placeholder(tf.float32)
#osc = single_oscillator(alpha, omega, beta1, beta2, delta1, delta2, eps, k, dt, x_in)

z = tf.Variable(tf.complex(z_init_r, z_init_i))
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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

r = []
for i in sin:
    r.append(sess.run(step, {x_in: i}))

sr = np.array(r).sum(axis=1)
plt.subplot(1,2,1)
plt.plot(sin)
plt.plot(sr)

plt.subplot(1,2,2)
ffsr = 1/sr.size * np.abs(np.fft.fft(sr)[1:sr.size//2])
ffreq = np.fft.fftfreq(sr.size)[1:sr.size//2]
plt.plot(ffreq, np.log(ffsr))
