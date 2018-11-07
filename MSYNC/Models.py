
import tensorflow as tf
import MSYNC.GFNN as GFNN


class BaseModel:
    def __init__(self, model_params):
        self.model_params = model_params
        self.v1_model = None
        self.v2_model = None
        self.dctw_model = None
        self.reg_model = None

    def build_single_branch_model(self):
        raise NotImplementedError

    def build_branch_models(self):
        self.v1_model = self.build_single_branch_model()
        self.v2_model = self.build_single_branch_model()
        return self.v1_model, self.v2_model

    def freeze_branch_models(self):
        for l1, l2 in zip(self.v1_model.layers, self.v2_model.layers):
            l1.trainable = l2.trainable = False

        self.v1_model = tf.keras.Model(self.v1_model.input, self.v1_model.output)
        self.v1_model = tf.keras.Model(self.v2_model.input, self.v2_model.output)
        return self.v1_model, self.v2_model

    def build_dctw_model(self):
        output = tf.keras.layers.concatenate([self.v1_model.output, self.v2_model.output], name='con_dctw')
        self.dctw_model = tf.keras.Model([self.v1_model.input, self.v2_model.input], output)
        return self.dctw_model

    def build_reg_model(self):
        cost_mat = tf.keras.layers.Lambda(cost_matrix_func, name='cost_mat')(self.dctw_model.output)
        output = tf.keras.layers.BatchNormalization()(cost_mat)
        output = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(output)
        output = tf.keras.layers.Conv2D(8, (5, 5), activation='relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(output)
        output = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(output)
        output = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(output)
        output = tf.keras.layers.Flatten()(output)
        output = tf.keras.layers.Dense(self.model_params['num_classes'], activation='linear')(output)
        self.reg_model = tf.keras.Model(self.dctw_model.input, output)
        return self.reg_model


class STFTModel(BaseModel):
    def __init__(self, model_params):
        super(STFTModel, self).__init__(model_params)

    def build_single_branch_model(self):
        # return self.build_lstm_branch_model()
        return self.build_dnn_branch_model()

    def build_lstm_branch_model(self):
        model_params = self.model_params
        input = tf.keras.Input(self.model_params['input_shape'])
        stft = tf.keras.layers.Lambda(lambda signal: stft_layer_func(signal, model_params))(input)

        output = tf.keras.layers.BatchNormalization()(stft)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2 * model_params['outdim_size'], activation='sigmoid'))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(output)

        model = tf.keras.Model(input, output)
        return model

    def build_dnn_branch_model(self):
        model_params = self.model_params
        input = tf.keras.Input(self.model_params['input_shape'])
        stft = tf.keras.layers.Lambda(lambda signal: stft_layer_func(signal, model_params))(input)

        output = tf.keras.layers.BatchNormalization()(stft)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(output))
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(output))
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='sigmoid'))(tf.keras.layers.BatchNormalization()(output))
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(model_params['outdim_size'], activation='linear'))(output)

        model = tf.keras.Model(input, output)
        return model


class GFNNModel(BaseModel):
    def __init__(self, model_params):
        super(GFNNModel, self).__init__(model_params)

    def build_single_branch_model(self):
        return self.build_lstm_branch_model()

    def build_lstm_branch_model(self):
        input = tf.keras.Input(self.model_params['input_shape'])
        gfnn_layer = GFNN.GFNNLayer(self.model_params['num_osc'], self.model_params['dt'], osc_params=self.model_params['osc_params'])
        gfnn = gfnn_layer(input)

        output = tf.keras.layers.BatchNormalization()(gfnn)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.model_params['outdim_size'], return_sequences=True))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2 * self.model_params['outdim_size'], activation='sigmoid'))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.model_params['outdim_size'], activation='linear'))(output)

        model = tf.keras.Model(input, output)
        return model


def stft_layer_func(signal, model_params):
    with tf.device('/device:GPU:0'):
        stft = tf.abs(tf.contrib.signal.stft(signal, model_params['stft_frame_length'], model_params['stft_frame_step'], pad_end=True))
        tf.summary.image('stft', tf.expand_dims(stft, -1))
    return stft


def cost_matrix_func(signals):
    def lin_norm(x, y):
        return tf.norm(x - y, axis=-1)

    def rkhs_norm(x, y, s=0.1):
        return tf.norm(gkernel(x, y, s), axis=-1)

    os = tf.shape(signals)[-1] // 2
    mat = tf.map_fn(lambda ri: lin_norm(tf.expand_dims(signals[:, ri, 0:os], axis=1), signals[:, :, os:os + os]),tf.range(tf.shape(signals)[1]), dtype=tf.float32)
    mat = tf.expand_dims(tf.transpose(mat, [1, 0, 2]), axis=-1)
    mat.set_shape([signals.get_shape().as_list()[0], signals.get_shape().as_list()[1], signals.get_shape().as_list()[1], 1])
    return mat


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )
