
import tensorflow as tf


def stft_layer_func(signal, model_params):
    with tf.device('/device:GPU:0'):
        stft = tf.abs(tf.contrib.signal.stft(signal, model_params['stft_frame_length'], model_params['stft_frame_step'], pad_end=True))
        tf.summary.image('stft', tf.expand_dims(stft, -1))
    return stft


def build_models(model_params):
    view1_in = tf.keras.Input(model_params['input_shape'])
    view2_in = tf.keras.Input(model_params['input_shape'])

    view1_middle_out, view1_end_out = build_stft_lstm_branch(view1_in, model_params)
    view2_middle_out, view2_end_out = build_stft_lstm_branch(view2_in, model_params)
    combined_output = tf.keras.layers.concatenate([view1_middle_out, view2_middle_out])

    class_output = tf.keras.layers.Dense(1024)(combined_output)
    class_output = tf.keras.layers.Dense(1024)(class_output)

    view1_model = tf.keras.Model(view1_in, view1_end_out)
    view2_model = tf.keras.Model(view2_in, view2_end_out)
    dctw_model = tf.keras.Model([view1_in, view2_in], combined_output)
    class_model = tf.keras.Model([view1_in, view2_in], class_output)
    return class_model, dctw_model, view1_model, view2_model


def build_stft_lstm_branch(input, model_params):
    stft = tf.keras.layers.Lambda(lambda signal: stft_layer_func(signal, model_params))(input)

    middle_output = tf.keras.layers.BatchNormalization()(stft)
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


def build_stft_dnn_branch(input, model_params):
    stft = tf.keras.layers.Lambda(lambda signal: stft_layer_func(signal, model_params))(input)

    middle_output = tf.keras.layers.BatchNormalization()(stft)
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(middle_output))
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(middle_output))
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(middle_output))
    middle_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(middle_output)

    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(middle_output)
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(end_output))
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(end_output))
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(end_output))
    end_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(tf.keras.layers.BatchNormalization()(end_output))
    end_output = tf.keras.layers.Flatten()(end_output)
    return middle_output, end_output
