
import tensorflow as tf
import numpy as np
import MSYNC.loss as loss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(loss)


y_pred = tf.convert_to_tensor(np.float32(np.random.randn(4, 1024, 256)))
# y_pred = np.float32(np.zeros([10, 20]))

r1 = 1e-4
r2 = 1e-4
eps = 1e-12
o1 = o2 = tf.shape(y_pred)[-1] // 2

# unpack (separate) the output of networks for view 1 and view 2
h1 = tf.transpose(y_pred[:, :, 0:o1], [0, 2, 1])
h2 = tf.transpose(y_pred[:, :, o1:o1 + o2], [0, 2, 1])

bs = tf.shape(h1)[0]
m = tf.shape(h1)[-1]
mf = tf.cast(m, tf.float32)

h1bar = h1 - (1.0 / mf) * h1
h2bar = h2 - (1.0 / mf) * h2

sigma_hat12 = (1.0 / (mf - 1)) * tf.matmul(h1bar, h2bar, transpose_b=True)
sigma_hat11 = (1.0 / (mf - 1)) * tf.matmul(h1bar, h1bar, transpose_b=True) + r1 * tf.eye(o1)
sigma_hat22 = (1.0 / (mf - 1)) * tf.matmul(h2bar, h2bar, transpose_b=True) + r2 * tf.eye(o2)

[d1, v1] = tf.linalg.eigh(sigma_hat11)
[d2, v2] = tf.linalg.eigh(sigma_hat22)

d1 = tf.where(d1 > eps, d1, eps * tf.ones_like(d1))
v1 = tf.where(v1 > eps, v1, eps * tf.ones_like(v1))
d2 = tf.where(d2 > eps, d2, eps * tf.ones_like(d2))
v2 = tf.where(v2 > eps, v2, eps * tf.ones_like(v2))

# mask1 = tf.map_fn(lambda val: val > eps, d1, dtype=tf.bool)
# mask2 = tf.map_fn(lambda val: val > eps, d2, dtype=tf.bool)
# mask = tf.logical_and(mask1, mask2)
#
# d1 = tf.map_fn(lambda bi: tf.boolean_mask(d1[bi, :], mask[bi, :]), tf.range(bs), dtype=tf.float32, infer_shape=False)
# v1 = tf.map_fn(lambda bi: tf.boolean_mask(v1[bi, :], mask[bi, :]), tf.range(bs), dtype=tf.float32, infer_shape=False)
# d2 = tf.map_fn(lambda bi: tf.boolean_mask(d2[bi, :], mask[bi, :]), tf.range(bs), dtype=tf.float32, infer_shape=False)
# v2 = tf.map_fn(lambda bi: tf.boolean_mask(v2[bi, :], mask[bi, :]), tf.range(bs), dtype=tf.float32, infer_shape=False)

sigma_hat11_root_inv = tf.matmul(tf.matmul(v1, tf.matrix_diag(tf.sqrt(d1))), v1, transpose_b=True)
sigma_hat22_root_inv = tf.matmul(tf.matmul(v2, tf.matrix_diag(tf.sqrt(d2))), v2, transpose_b=True)

tval = tf.matmul(tf.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv)

corr = tf.sqrt(tf.linalg.trace(tf.matmul(tval, tval, transpose_b=True)))
corr = tf.reduce_mean(corr)

sess = tf.Session()
r = sess.run(corr)

