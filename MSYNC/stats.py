
import tensorflow as tf
import numpy as np
import tfmpl


@tfmpl.figure_tensor
def create_ave_image(ecl_distance):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ecl_distance)
    # ax.set_title('dist = ' + str(dist))
    return fig


class TensorBoardAVE(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardAVE, self).__init__(**kwargs)

    def _make_histogram_ops(self, model):
        super(TensorBoardAVE, self)._make_histogram_ops(model)
        tf.summary.image('sequence_ecl_distance', create_ave_image(model.get_layer('EclDistance').output))
