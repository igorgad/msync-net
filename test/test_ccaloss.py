
import tensorflow as tf
import numpy as np
import models.loss as loss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(loss)


y_pred = np.float32(np.random.randn(10, 20))
# y_pred = np.float32(np.zeros([10, 20]))

cca_loss = loss.cca_loss(y_pred, y_pred)

sess = tf.Session()
r = sess.run(cca_loss)

