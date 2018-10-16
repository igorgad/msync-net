
import tensorflow as tf
import numpy as np
import MSYNC.loss as loss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(loss)


y_pred = tf.convert_to_tensor(np.float32(np.random.randn(4, 10, 20)))
# y_pred = np.float32(np.zeros([10, 20]))


r1 = 1e-4
r2 = 1e-4
eps = 1e-12
o1 = o2 = tf.shape(y_pred)[-1] // 2

# unpack (separate) the output of networks for view 1 and view 2
h1 = tf.transpose(y_pred[:, :, 0:o1], [0, 2, 1])
h2 = tf.transpose(y_pred[:, :, o1:o1 + o2], [0, 2, 1])

m = tf.shape(h1)[-1]
mf = tf.cast(m, tf.float32)

h1bar = h1 - (1.0 / mf) * h1
h2bar = h2 - (1.0 / mf) * h2

sigma_hat12 = (1.0 / (mf - 1)) * tf.matmul(h1bar, tf.transpose(h2bar, [0, 2, 1]))
sigma_hat11 = (1.0 / (mf - 1)) * tf.matmul(h1bar, tf.transpose(h1bar, [0, 2, 1])) + r1 * tf.eye(o1)
sigma_hat22 = (1.0 / (mf - 1)) * tf.matmul(h2bar, tf.transpose(h2bar, [0, 2, 1])) + r2 * tf.eye(o2)

sigma_hat11_root_inv = tf.matrix_inverse(sigma_hat11)
sigma_hat22_root_inv = tf.matrix_inverse(sigma_hat22)

tval = tf.matmul(tf.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv)

corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(tval, [0, 2, 1]), tval)))
corr = tf.reduce_mean(corr)

sess = tf.Session()
r = sess.run(corr)

