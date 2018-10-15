
import tensorflow as tf
import MSYNC.GFNN as GFNN
import MSYNC.loss as loss


def simple_gfnn_cca_v0(model_params):
    view1_in = tf.keras.Input(model_params['input_shape'])
    view2_in = tf.keras.Input(model_params['input_shape'])

    view1_middle_out, view1_end_out = build_gfnn_lstm_branch(view1_in, model_params)
    view2_middle_out, view2_end_out = build_gfnn_lstm_branch(view2_in, model_params)

    view1_model = tf.keras.Model(view1_in, view1_end_out)
    view2_model = tf.keras.Model(view2_in, view2_end_out)

    combined_output = tf.keras.layers.concatenate([view1_model.layers[-1].output, view2_model.layers[-1].output])

    model = tf.keras.Model([view1_in, view2_in], combined_output)

    view1_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['lr']))
    view2_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['lr']))
    model.compile(loss=loss.cca_loss, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['lr']))

    return model, view1_model, view2_model


def build_gfnn_lstm_branch(input, model_params):
    gfnn = GFNN.GFNNLayer(model_params['num_osc'], model_params['dt'], osc_params=model_params['osc_params'])
    middle_output = gfnn(input)
    middle_output = tf.keras.layers.LSTM(model_params['outdim_size'])(middle_output)
    end_output = tf.keras.layers.Dense(model_params['input_shape'][0])(middle_output)
    return middle_output, end_output
