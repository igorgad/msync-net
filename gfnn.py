# Author - Igor Pereira
# UFRN - ISASI

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 8, dt)
ff = 6
sin = 0.7 * np.exp(1j * 2 * np.pi * ff * t)
#sin = np.concatenate([0.25 * np.sin(t[:t.size//2]*1*np.pi),  np.zeros(t.size//2)])

nosc = 128
fmin = 0.125
fmax = 8.0
f = np.logspace(np.log10(fmin), np.log10(fmax), nosc, dtype=np.float32)
w = 2 * np.pi * f
per = np.floor(len(f)/(np.log2(f[-1])-np.log2(f[0])))
alpha = 0.0 * np.ones_like(f)
beta1 = -100.0
beta2 = 0.0
delta1 = 0.0
delta2 = 0.0
eps = np.complex64(1.0)
k = np.complex64(0.0)

lamb = np.complex64(0.1 + 0j) #0.001
mu1 = np.complex64(-1.0 + 0.0j) #-1.0
mu2 = np.complex64(-50.0 + 0.0j) #-50.0
epsc = np.complex64(1.0 + 0j)
kc = np.complex64(1.0 + 0j)

c_init = np.complex64(np.random.standard_normal([nosc, nosc]) * 0.0)

r0 = 0 + 0.00 * np.random.standard_normal(nosc)
phi0 = 0 * np.pi * np.random.standard_normal(nosc)
z_init = np.complex64(r0 * np.exp(1j * phi0, dtype=np.complex64))

with tf.device('/device:GPU:0'):
    x_in = tf.placeholder(tf.complex64)
    z = tf.Variable(z_init)
    c = tf.Variable(c_init)

    ints = tf.squeeze(tf.matmul(tf.matrix_set_diag(c, tf.zeros_like(tf.diag_part(c))), tf.expand_dims(z, axis=1)), axis=1)
    x = x_in + k * ints

    # a = tf.complex(alpha, 2*np.pi*f)
    # b = tf.complex(beta1, delta1)
    # d = tf.complex(beta2, delta2)
    #
    a = (alpha + 1j * 2 * np.pi) * f
    b = (beta1 + 1j * delta1) * f
    d = (beta2 + 1j * delta2) * f
    z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
    z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
    z_ = tf.conj(z)
    passive = x * tf.reciprocal(1.0 - tf.sqrt(eps) * x)
    active = tf.reciprocal(1.0 - tf.sqrt(eps) * z_)

    dzdt = dt * (tf.multiply(z, a + b * z2 + d * eps * z4 / (1 - eps * z2)) + active * passive * f)
    dzdt = tf.where(tf.is_nan(tf.real(dzdt)), tf.zeros_like(dzdt), dzdt)
    dzdt = tf.where(tf.is_inf(tf.real(dzdt)), tf.ones_like(dzdt), dzdt)
    z_step = z.assign((z + dzdt))

    #######################################################
    def zfunc(zi, zj):
        return zi * tf.reciprocal(1 - tf.sqrt(epsc) * zi) * zj * tf.reciprocal(1 - tf.sqrt(epsc) * tf.conj(zj)) \
               * tf.reciprocal(1 - tf.sqrt(epsc) * zj)


    #fzz = tf.map_fn(lambda i: tf.map_fn(lambda j: zfunc(z[i], z[j]), tf.range(nosc), dtype=tf.complex64), tf.range(nosc), dtype=tf.complex64)
    fzz = f * tf.map_fn(lambda i: zfunc(z[i], z), tf.range(nosc), dtype=tf.complex64)
    dcdt = dt * (c * (lamb + mu1 * tf.complex(tf.pow(tf.abs(c), 2), 0.0) + tf.divide(epsc * mu2 * tf.complex(tf.pow(tf.abs(c), 4), 0.0),
                                                                                 1 - epsc * tf.complex(tf.pow(tf.abs(c), 2), 0.0))) + kc * fzz )

    dcdt = tf.where(tf.is_nan(tf.real(dcdt)), tf.zeros_like(dcdt), dcdt)
    dcdt = tf.where(tf.is_inf(tf.real(dcdt)), tf.ones_like(dcdt), dcdt)
    c_step = c.assign(c + dcdt)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

pp = []
aa = []
r = []
cr = []
for i in sin:
    r.append(sess.run(z_step, {x_in: i}))
    pp.append(sess.run(passive, {x_in: i}))
    aa.append(sess.run(active, {x_in: i}))
    cr.append(sess.run(c_step, {x_in: i}))


sr = np.array(r)
sp = np.array(passive)
sa = np.array(active)
sc = np.array(cr)

freq = f
xticks = np.arange(0, nosc, 50)
xfreq = freq[range(0, nosc, 50)]
xlabels = ["%.1f" % x for x in xfreq]

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.imshow(np.real(sr).T)
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax2.imshow(np.angle(sr.T), cmap='gray')
ax2.set_yticks(xticks)
ax2.set_yticklabels(xlabels)

ax3.plot(np.real(sr).mean(0))
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)


