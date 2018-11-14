
import tensorflow as tf
import numpy as np
import MSYNC.utils as utils
import dataset_interface
import matplotlib.pyplot as plt


pred = np.float32(np.zeros([4, 16]))
pred[0, 12] = 2
for i in range(1, pred.shape[0]):
    pred[i, np.random.randint(16)] = 2

true = np.float32(np.zeros([4, 16]))
true[0, 10] = 1
for i in range(1, pred.shape[0]):
    true[i, np.random.randint(16)] = 1

acc = utils.range_categorical_accuracy(true, pred)

sess = tf.Session()
r = sess.run(acc)
