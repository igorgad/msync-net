
import tensorflow as tf


def cca_loss(y_true, y_pred):
    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12
    o1 = o2 = y_pred.shape[1] // 2

    # unpack (separate) the output of networks for view 1 and view 2
    H1 = y_pred[:, 0:o1].T
    H2 = y_pred[:, o1:o1 + o2].T

    m = H1.shape[1]

    H1bar = H1 - (1.0 / m) * tf.matmul(H1, tf.ones([m, m]))
    H2bar = H2 - (1.0 / m) * tf.matmul(H2, tf.ones([m, m]))

    SigmaHat12 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H2bar, [0, 1]))
    SigmaHat11 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H1bar, [0, 1])) + r1 * tf.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * tf.matmul(H2bar, tf.transpose(H2bar, [0, 1])) + r2 * tf.eye(o2)

    SigmaHat11RootInv = tf.matrix_inverse(SigmaHat11)
    SigmaHat22RootInv = tf.matrix_inverse(SigmaHat22)

    Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval, [0, 1]), Tval)))

    return -corr
