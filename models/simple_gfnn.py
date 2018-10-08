
import tensorflow as tf
import numpy as np
import models.GFNN as GFNN
import models.loss as loss


def simple_gfnn_cca_v0(model_params):

    view1_model = build_single_branch(model_params)
    view2_model = build_single_branch(model_params)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.concatenate([view1_model, view2_model]))

    model_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['lr'])
    model.compile(loss=loss.cca_loss(model_params['outdim_size'], model_params['use_all_singular_values']), optimizer=model_optimizer)

    return model


def build_single_branch(model_params):
    gfnn = GFNN.KerasLayer(model_params['num_osc'], model_params['dt'])

    model = tf.keras.Sequential()
    model.add(gfnn)
    model.add(tf.keras.layers.LSTM(32))

    return model
