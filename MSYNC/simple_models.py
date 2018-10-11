
import tensorflow as tf
import MSYNC.GFNN as GFNN
import MSYNC.loss as loss
from kapre.time_frequency import Spectrogram


def simple_gfnn_cca_v0(model_params):

    view1_in, view1_out = build_gfnn_lstm_branch(model_params)
    view2_in, view2_out = build_gfnn_lstm_branch(model_params)

    combined_output = tf.keras.layers.concatenate([view1_out, view2_out])

    model = tf.keras.Model([view1_in, view2_in], combined_output)
    model_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['lr'])
    model.compile(loss=loss.cca_loss, optimizer=model_optimizer)

    return model


def build_gfnn_lstm_branch(model_params):
    gfnn = GFNN.GFNNLayer(model_params['num_osc'], model_params['dt'])

    input = tf.keras.Input(model_params['input_shape'])
    output = gfnn(input)
    output = tf.keras.layers.LSTM(model_params['outdim_size'])(output)
    return input, output


def build_stft_lstm_branch(model_params):
    stft = Spectrogram(n_dft=512, n_hop=model_params['num_osc'], return_decibel_spectrogram=True, power_spectrogram=2.0, name='static_stft')

    input = tf.keras.Input(model_params['input_shape'])
    output = stft(input)
    output = tf.keras.layers.LSTM(model_params['outdim_size'])(output)
    return input, output
