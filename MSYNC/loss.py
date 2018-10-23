
import tensorflow as tf


def compute_batch_loss(y_pred, outdim_size, use_all_singular_values):
    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12
    o1 = o2 = tf.shape(y_pred)[-1] // 2

    # unpack (separate) the output of networks for view 1 and view 2
    h1 = tf.transpose(y_pred[:, 0:o1])
    h2 = tf.transpose(y_pred[:, o1:o1 + o2])

    m = tf.shape(h1)[-1]
    mf = tf.cast(m, tf.float32)

    h1bar = h1 - (1.0 / mf) * h1
    h2bar = h2 - (1.0 / mf) * h2

    sigma_hat12 = (1.0 / (mf - 1)) * tf.matmul(h1bar, h2bar, transpose_b=True)
    sigma_hat11 = (1.0 / (mf - 1)) * tf.matmul(h1bar, h1bar, transpose_b=True) + r1 * tf.eye(o1)
    sigma_hat22 = (1.0 / (mf - 1)) * tf.matmul(h2bar, h2bar, transpose_b=True) + r2 * tf.eye(o2)

    [d1, v1] = tf.linalg.eigh(sigma_hat11)
    [d2, v2] = tf.linalg.eigh(sigma_hat22)

    d1_idx = tf.where(tf.greater(d1, eps))
    d1 = tf.gather_nd(d1, d1_idx)
    v1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(v1), tf.squeeze(d1_idx)))

    d2_idx = tf.where(tf.greater(d2, eps))
    d2 = tf.gather_nd(d2, d2_idx)
    v2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(v2), tf.squeeze(d2_idx)))

    sigma_hat11_root_inv = tf.matmul(tf.matmul(v1, tf.diag(d1 ** -0.5)), v1, transpose_b=True)
    sigma_hat22_root_inv = tf.matmul(tf.matmul(v2, tf.diag(d2 ** -0.5)), v2, transpose_b=True)

    tval = tf.matmul(tf.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv)

    if use_all_singular_values:
        corr = tf.sqrt(tf.trace(tf.matmul(tval, tval, transpose_a=True)))
    else:
        [u, v] = tf.self_adjoint_eig(tf.matmul(tval, tval, transpose_a=True))
        u = tf.gather_nd(u, tf.where(tf.greater(u, eps)))
        kk = tf.reshape(tf.cast(tf.shape(u), tf.int32), [])
        K = tf.minimum(kk, outdim_size)
        w, _ = tf.nn.top_k(u, k=K)
        corr = tf.reduce_sum(tf.sqrt(w))

    return -corr


def cca_loss(outdim_size, use_all_singular_values):
    def inner_cca_loss(y_true, y_pred):
        batch_corr = tf.map_fn(lambda bi: compute_batch_loss(y_pred[bi], outdim_size, use_all_singular_values), tf.range(tf.shape(y_pred)[0]), dtype=tf.float32)
        return tf.reduce_mean(batch_corr)

    return inner_cca_loss
