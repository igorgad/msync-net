
import tensorflow as tf
import numpy as np
import tfmpl


@tfmpl.figure_tensor
def create_ave_image(ecl_distance, targets):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_ylim(0.0, 1.0)
    ax.plot(np.arange(-ecl_distance.shape[1] // 2, ecl_distance.shape[1] // 2), ecl_distance[0,:])
    ax.axvline(np.argmax(targets[0,:]) - targets[0,:].size//2)
    # ax.set_title('dist = ' + str(dist))
    return fig


class TensorBoardAVE(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardAVE, self).__init__(**kwargs)

    def _make_histogram_ops(self, model):
        super(TensorBoardAVE, self)._make_histogram_ops(model)
        tf.summary.image('sequence_ecl_distance', create_ave_image(model.get_layer('ecl_distance').output, model.targets[0]))
