

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

nosc = 90
batch_size = 1

Fs = 60.0
dt = 1.0/Fs
t = np.arange(0, 2, dt)
ff = 2.0
sin = np.complex64(0.25 * np.exp(1j * 2 * np.pi * ff * t))
sin = np.repeat(np.expand_dims(sin, 0),  batch_size, axis=0)

fmin = 0.5
fmax = 8.0
f = np.logspace(np.log10(fmin), np.log10(fmax), nosc, dtype=np.float32)
w = 2 * np.pi * f
per = np.floor(len(f)/(np.log2(f[-1])-np.log2(f[0])))
alpha = -1.0 * np.ones_like(f)
beta1 = -50.0
beta2 = 0.0
delta1 = 0.0
delta2 = 0.0
eps = np.complex64(1.0)
k = np.complex64(0.0)

a = (alpha + 1j * 2 * np.pi) * f
b = (beta1 + 1j * delta1) * f
d = (beta2 + 1j * delta2) * f

lamb = np.complex64(0.000 + 0j) #0.001
mu1 = np.complex64(-1.0 + 0.0j) #-1.0
mu2 = np.complex64(-50.0 + 0.0j) #-50.0
epsc = np.complex64(1.0 + 0j)
kc = np.complex64(1.0 + 0j)

r0 = 0 + 0.00 * np.random.standard_normal([batch_size, nosc])
phi0 = 0 * np.pi * np.random.standard_normal([batch_size, nosc])
z_init = np.complex64(r0 * np.exp(1j * phi0, dtype=np.complex64))
c_init = np.complex64(np.random.standard_normal([batch_size, nosc, nosc]) * 0.0).reshape([batch_size, -1])

init_state = np.concatenate([z_init, c_init], -1)


def zdot(state, ti, nosc):
    z = tf.gather(state, tf.range(nosc), axis=-1)
    c = tf.gather(state, tf.range(nosc, nosc * nosc + nosc), axis=-1)
    c = tf.reshape(c, [-1, nosc, nosc])

    x = tf.expand_dims(tf.gather(sin, ti, axis=-1), axis=-1) + tf.squeeze(tf.matmul(c, tf.expand_dims(z, axis=-1)), axis=-1)

    # Oscillator Update Rule
    z2 = tf.complex(tf.pow(tf.abs(z), 2), 0.0)
    z4 = tf.complex(tf.pow(tf.abs(z), 4), 0.0)
    z_ = tf.conj(z)
    passive = x * tf.reciprocal(1.0 - tf.sqrt(eps) * x)
    active = tf.reciprocal(1.0 - tf.sqrt(eps) * z_)
    dzdt = tf.multiply(z, a + b * z2 + d * eps * z4 / (1 - eps * z2)) + active * passive * f
    dcdt = c

    # Connectivity Update Rule
    # def zmul(zi, zj):
    #     return zi * tf.reciprocal(1 - tf.sqrt(epsc) * zi) * zj * tf.reciprocal(1 - tf.sqrt(epsc) * tf.conj(zj)) \
    #            * tf.reciprocal(1 - tf.sqrt(epsc) * zj)
    #
    # fzz = tf.transpose(tf.map_fn(lambda i: zmul(tf.expand_dims(z[:,i], axis=-1), z), tf.range(nosc), dtype=tf.complex64), [1, 0, 2])
    # dcdt = c * (lamb + mu1 * tf.complex(tf.pow(tf.abs(c), 2), 0.0) + tf.divide(epsc * mu2 * tf.complex(tf.pow(tf.abs(c), 4), 0.0), 1 - epsc * tf.complex(tf.pow(tf.abs(c), 2), 0.0))) + kc * fzz * f
    #
    # dcdt = tf.where(tf.is_nan(tf.real(dcdt)), tf.zeros_like(dcdt), dcdt)
    # dcdt = tf.where(tf.is_inf(tf.real(dcdt)), tf.ones_like(dcdt), dcdt)

    return tf.concat([dzdt, tf.reshape(dcdt, [-1, nosc * nosc])], axis=-1)


with tf.device('/device:GPU:1'):
    input = sin
    batch_size = tf.shape(input)[0]
    input_size = tf.shape(input)[1]

    dtt = tf.convert_to_tensor(dt)
    t = tf.range(0, tf.cast(input_size, tf.float32) * dtt, dt)

    tensor_state = tf.contrib.integrate.odeint_fixed(lambda s, t: zdot(s, tf.cast(t // dtt, tf.int32), nosc), init_state, t, dtt, method='rk4')
    tensor_state = tf.transpose(tensor_state, [1, 0, 2])
    z_state = tf.gather(tensor_state, tf.range(nosc), axis=-1)
    c_state = tf.gather(tensor_state, tf.range(nosc, nosc * nosc + nosc), axis=-1)
    c_state = tf.reshape(c_state, [batch_size, -1, nosc, nosc])


config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

st = time.time()
sr, sc = sess.run([z_state, c_state])
print (time.time() - st)

freq = f
xticks = np.arange(0, nosc, 50)
xfreq = freq[range(0, nosc, 50)]
xlabels = ["%.1f" % x for x in xfreq]

fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(14, 8))
ax1.imshow(np.real(sr[0,:,:]).T, cmap='gray')
ax1.set_yticks(xticks)
ax1.set_yticklabels(xlabels)

ax2.imshow(np.angle(sr[0,:,:]).T, cmap='gray')
ax2.set_yticks(xticks)
ax2.set_yticklabels(xlabels)

ax3.imshow(np.real(sc[0,-1,:,:]).T, cmap='gray')
ax3.set_xticks(xticks)
ax3.set_xticklabels(xlabels)
ax3.set_yticks(xticks)
ax3.set_yticklabels(xlabels)
