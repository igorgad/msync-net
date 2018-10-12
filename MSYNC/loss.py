
import tensorflow as tf


def cca_loss(y_true, y_pred):
    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12
    o1 = o2 = tf.shape(y_pred)[1] // 2

    # unpack (separate) the output of networks for view 1 and view 2
    h1 = tf.transpose(y_pred[:, 0:o1], [1, 0])
    h2 = tf.transpose(y_pred[:, o1:o1 + o2], [1, 0])

    m = tf.shape(h1)[1]
    mf = tf.cast(m, tf.float32)

    h1bar = h1 - (1.0 / mf) * tf.matmul(h1, tf.ones([m, m]))
    h2bar = h2 - (1.0 / mf) * tf.matmul(h2, tf.ones([m, m]))

    sigma_hat12 = (1.0 / (mf - 1)) * tf.matmul(h1bar, tf.transpose(h2bar, [1, 0]))
    sigma_hat11 = (1.0 / (mf - 1)) * tf.matmul(h1bar, tf.transpose(h1bar, [1, 0])) + r1 * tf.eye(o1)
    sigma_hat22 = (1.0 / (mf - 1)) * tf.matmul(h2bar, tf.transpose(h2bar, [1, 0])) + r2 * tf.eye(o2)

    sigma_hat11_root_inv = tf.matrix_inverse(sigma_hat11)
    sigma_hat22_root_inv = tf.matrix_inverse(sigma_hat22)

    tval = tf.matmul(tf.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv)

    corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(tval, [1, 0]), tval)))
    return -corr
