
import tensorflow as tf


def vgg_encoder(input, trainable=True, name=''):
    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block1/conv1')(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_encoder_block1/pool')(x)

    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block2/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_encoder_block2/pool')(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block3/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block3/conv2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_encoder_block3/pool')(x)

    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block4/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_encoder_block4/conv2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_encoder_block4/pool')(x)

    return x


def vgg_decoder(input, trainable=True, name=''):
    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name + 'vgg_decoder_block4/conv1')(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name + 'vgg_decoder_block4/conv2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)), trainable=trainable, name=name + 'vgg_decoder_block4/pool')(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name + 'vgg_decoder_block3/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name + 'vgg_decoder_block3/conv2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)), trainable=trainable, name=name + 'vgg_decoder_block3/pool')(x)

    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name + 'vgg_decoder_block2/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)), trainable=trainable, name=name + 'vgg_decoder_block2/pool')(x)

    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_decoder_block1/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)), trainable=trainable, name=name+'vgg_decoder_block1/pool')(x)
    
    # Block 0
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same'), trainable=trainable, name=name+'vgg_decoder_block0/conv1')(x)

    return x

