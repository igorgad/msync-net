
import tensorflow as tf
import MSYNC.GFNN as GFNN


def build_models(model_params):
    view1_in = tf.keras.Input(model_params['input_shape'])
    view2_in = tf.keras.Input(model_params['input_shape'])

    view1_middle_out, view1_end_out = build_gfnn_lstm_branch(view1_in, model_params)
    view2_middle_out, view2_end_out = build_gfnn_lstm_branch(view2_in, model_params)
    combined_output = tf.keras.layers.concatenate([view1_middle_out, view2_middle_out])

    class_output = tf.keras.layers.BatchNormalization()(combined_output)
    class_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(2 * model_params['outdim_size'], return_sequences=True))(class_output)
    class_output = tf.keras.layers.BatchNormalization()(class_output)
    class_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(2 * model_params['outdim_size']))(class_output)
    class_output = tf.keras.layers.BatchNormalization()(class_output)
    class_output = tf.keras.layers.Dense(model_params['num_classes'], activation='softmax')(class_output)

    view1_model = tf.keras.Model(view1_in, view1_end_out)
    view2_model = tf.keras.Model(view2_in, view2_end_out)
    model = tf.keras.Model([view1_in, view2_in], combined_output)
    class_model = tf.keras.Model([view1_in, view2_in], class_output)
    return class_model, model, view1_model, view2_model


def build_gfnn_lstm_branch(input, model_params):
    gfnn_layer = GFNN.GFNNLayer(model_params['num_osc'], model_params['dt'], osc_params=model_params['osc_params'])
    gfnn = gfnn_layer(input)

    middle_output = tf.keras.layers.BatchNormalization()(gfnn)
    middle_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(middle_output)
    middle_output = tf.keras.layers.BatchNormalization()(middle_output)
    middle_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(middle_output)
    middle_output = tf.keras.layers.BatchNormalization()(middle_output)
    middle_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(middle_output)
    middle_output = tf.keras.layers.BatchNormalization()(middle_output)
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2 * model_params['outdim_size'], activation='sigmoid'))(middle_output)
    middle_output = tf.keras.layers.BatchNormalization()(middle_output)
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(middle_output)

    end_output = tf.keras.layers.BatchNormalization()(middle_output)
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(end_output)
    end_output = tf.keras.layers.BatchNormalization()(end_output)
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2 * model_params['outdim_size'], activation='sigmoid'))(end_output)
    end_output = tf.keras.layers.BatchNormalization()(end_output)
    end_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(end_output)
    end_output = tf.keras.layers.BatchNormalization()(end_output)
    end_output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(end_output)
    end_output = tf.keras.layers.BatchNormalization()(end_output)
    end_output = tf.keras.layers.CuDNNLSTM(1, return_sequences=True)(end_output)
    end_output = tf.keras.layers.Flatten()(end_output)
    return middle_output, end_output
