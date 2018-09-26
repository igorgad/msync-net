# Author - Igor Pereira
# UFRN - ISASI

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 20, dt)
ff = 2
sin = 0.25 * np.sin(t*2*np.pi*ff)
#sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*1*np.pi),  np.zeros(t.size//2)])

nosc = 360
omega = 2 * np.pi * np.logspace(0, 1, nosc, dtype=np.float32) / 10
alpha = -0.5 * np.ones(nosc, np.float32)
beta1 = -10.0
beta2 = -9.0
delta1 = -10.0
delta2 = -9.0
eps = np.complex64(1.0 + 0j)
k = np.complex64(1.0 + 0j)

c_init = 0.00 * np.ones([nosc, nosc], dtype=np.complex64)
z_init = 0.00 * np.ones(nosc, dtype=np.complex64)

with tf.device('/device:GPU:1'):
    x_in = tf.placeholder(tf.float32)
    z = tf.Variable(z_init)
    c = tf.Variable(c_init)

    ints = tf.squeeze(tf.matmul(c, tf.expand_dims(z, axis=1)), axis=1)
    x = tf.complex(x_in, 0.0) + k * ints

    a = tf.complex(alpha, omega)
    b = tf.complex(beta1, delta1)
    d = tf.complex(beta2, delta2)
    z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
    z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
    z_ = tf.conj(z)
    passive = x * tf.reciprocal(1.0 - tf.sqrt(eps) * x)
    active = tf.reciprocal(1.0 - tf.sqrt(eps) * z_)

    dzdt = dt * (tf.multiply(z, a + b * z2 + d * eps * z4 / (1 - eps * z2)) + passive * active)
    z_step = z.assign(z + dzdt)


    lamb = np.complex64(0.01 + 0j) #0.001
    mu1 = np.complex64(-1.0 + 0.0j) #-1.0
    mu2 = np.complex64(-50.0 + 0.0j) #-50.0
    epsc = np.complex64(1.0 + 0j)
    kc = np.complex64(1.0 + 0j)

    def zfunc(zi, zj):
        return zi * tf.reciprocal(1 - tf.sqrt(epsc) * zi) * zj * tf.reciprocal(1 - tf.sqrt(epsc) * tf.conj(zj)) \
               * tf.reciprocal(1 - tf.sqrt(epsc) * zj)


    #fzz = tf.map_fn(lambda i: tf.map_fn(lambda j: zfunc(z[i], z[j]), tf.range(nosc), dtype=tf.complex64), tf.range(nosc), dtype=tf.complex64)
    fzz = tf.map_fn(lambda i: zfunc(z[i], z), tf.range(nosc), dtype=tf.complex64)
    dcdt = dt * (c * (lamb + mu1 * tf.complex(tf.pow(tf.abs(c), 2), 0.0) + tf.divide(epsc * mu2 * tf.complex(tf.pow(tf.abs(c), 4), 0.0),
                                                                                 1 - epsc * tf.complex(tf.pow(tf.abs(c), 2), 0.0))) + kc * fzz )

    c_step = c.assign(c + dcdt)


config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

r = []
cr = []
for i in sin:
    r.append(sess.run(z_step, {x_in: i}))
    cr.append(sess.run(c_step, {x_in: i}))


sr = np.array(r).sum(axis=1)
sc = np.array(cr)

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.plot(sin)
ax1.plot(sr)

n = sr.size
T = n * dt
freq = np.arange(n)[1:n//2]/T
logfreq = np.log(freq)
xticks = np.linspace(logfreq[0], logfreq[-1], 10)

ffsr = 1/n * np.abs(np.fft.fft(sr)[1:n//2])
ax2.plot(logfreq, np.log(ffsr))
ax2.set_xticks(xticks)
ax2.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])

ax3.imshow(np.abs(sc[-1,:,:]))