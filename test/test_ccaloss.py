
import tensorflow as tf
import numpy as np
import models.loss as loss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(loss)


y_pred = tf.convert_to_tensor(np.float32(np.random.randn(10, 20)))
# y_pred = np.float32(np.zeros([10, 20]))


r1 = 1e-4
r2 = 1e-4
eps = 1e-12
o1 = o2 = tf.shape(y_pred)[1] // 2

# unpack (separate) the output of networks for view 1 and view 2
H1 = tf.transpose(y_pred[:, 0:o1], [1, 0])
H2 = tf.transpose(y_pred[:, o1:o1 + o2], [1, 0])

m = tf.shape(H1)[1]
mf = tf.cast(m, tf.float32)

H1bar = H1 - (1.0 / mf) * tf.matmul(H1, tf.ones([m, m]))
H2bar = H2 - (1.0 / mf) * tf.matmul(H2, tf.ones([m, m]))

SigmaHat12 = (1.0 / (mf - 1)) * tf.matmul(H1bar, tf.transpose(H2bar, [1, 0]))
SigmaHat11 = (1.0 / (mf - 1)) * tf.matmul(H1bar, tf.transpose(H1bar, [1, 0])) + r1 * tf.eye(o1)
SigmaHat22 = (1.0 / (mf - 1)) * tf.matmul(H2bar, tf.transpose(H2bar, [1, 0])) + r2 * tf.eye(o2)

SigmaHat11RootInv = tf.matrix_inverse(SigmaHat11)
SigmaHat22RootInv = tf.matrix_inverse(SigmaHat22)

Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

corr = -1 * tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval, [1, 0]), Tval)))

sess = tf.Session()
r = sess.run(corr)

