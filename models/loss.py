
import tensorflow as tf


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_cca_objective(y_true, y_pred):

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = y_pred.shape[1]//2

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = y_pred[:, 0:o1].T
        H2 = y_pred[:, o1:o1+o2].T

        m = H1.shape[1]

        H1bar = H1 - (1.0 / m) * tf.matmul(H1, tf.ones([m, m]))
        H2bar = H2 - (1.0 / m) * tf.matmul(H2, tf.ones([m, m]))

        SigmaHat12 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H2bar, [0, 2, 1]))
        SigmaHat11 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H1bar, [0, 2, 1])) + r1 * tf.eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * tf.matmul(H2bar, tf.transpose(H2bar, [0, 2, 1])) + r2 * tf.eye(o2)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.linalg.eigh(SigmaHat11)
        [D2, V2] = tf.linalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = tf.where(tf.greater(D1, eps))[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = tf.where(tf.greater(D2, eps))[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1 ** -0.5)), tf.transpose(V1, [0, 2, 1]))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2 ** -0.5)), tf.transpose(V2, [0, 2, 1]))

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval, [0, 2, 1]), Tval)))
        else:
            # just the top outdim_size singular values are used
            [U, V] = tf.linalg.eigh(tf.matmul(tf.transpose(Tval, [0, 2, 1]), Tval))
            U = U[tf.where(tf.greater(U, eps))[0]]
            U = tf.contrib.framework.sort(U)
            corr = tf.reduce_sum(tf.sqrt(U[0:outdim_size]))

        return -corr

    return inner_cca_objective
