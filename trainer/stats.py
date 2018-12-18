
import tensorflow as tf
import numpy as np
import tfmpl


@tfmpl.figure_tensor
def create_ave_image(ecl_distance, targets):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(-ecl_distance.shape[1] //2 + 1, ecl_distance.shape[1] // 2 + 1), ecl_distance[0,:])
    ax.plot(np.arange(-ecl_distance.shape[1] // 2 + 1, ecl_distance.shape[1] // 2 + 1), targets[0, :])
    # ax.axvline(np.argmax(targets[0, :]) - targets[0, :].size // 2)
    # ax.axvline(np.argmax(targets[0, :]) - targets[0, :].size // 2)
    return fig


@tfmpl.figure_tensor
def create_inputs_plot(i1, i2):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(i1[0].reshape(-1))
    ax.plot(i2[0].reshape(-1))
    return fig


class TensorBoardAVE(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardAVE, self).__init__(**kwargs)

    def _make_histogram_ops(self, model):
#         super(TensorBoardAVE, self)._make_histogram_ops(model)
        tf.summary.image('sequence_ecl_distance', create_ave_image(model.get_layer('diag_mean').output, model.targets[0]))
        # tf.summary.image('input_plots', create_inputs_plot(model.inputs[0], model.inputs[1]))

        i1_audio = tf.expand_dims(tf.reshape(model.inputs[0], [-1, 8 * 15360]), axis=-1)
        i2_audio = tf.expand_dims(tf.reshape(model.inputs[1], [-1, 8 * 15360]), axis=-1)
        tf.summary.audio('input1_audio', i1_audio, 16000)
        tf.summary.audio('input2_audio', i2_audio, 16000)
        tf.summary.audio('mixed_audio', tf.reduce_mean(tf.concat([i1_audio, i2_audio], axis=-1), axis=-1), 16000)
