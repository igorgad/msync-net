
import tensorflow as tf
import numpy as np
from dtw import fastdtw
import tfmpl


@tfmpl.figure_tensor
def compute_dtw_and_create_image(r):
    idx = np.arange(0, r.shape[1], 64)
    r1 = r[0, idx, :r.shape[-1] // 2]
    r2 = r[0, idx, r.shape[-1] // 2:]
    dist, cost, acc_cost, path = fastdtw(r1, r2, dist=lambda x, y: np.linalg.norm(np.power(x - y, 2), ord=1))

    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
    ax.plot(path[0], path[1], 'w')
    ax.set_title('dist = ' + str(dist))
    return fig


class TensorBoardDTW(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardDTW, self).__init__(**kwargs)
        self.dtw_image_summary = None

    def _make_histogram_ops(self, model):
        super(TensorBoardDTW, self)._make_histogram_ops(model)
        tf.summary.image('dtw-cost', compute_dtw_and_create_image(model.get_layer('con_dctw').output))
