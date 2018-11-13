
import tensorflow as tf

# weight path
WEIGHTS_PATH = './saved_models/vggish_audioset_weights_without_fc2.h5'


def vggish(input, trainable=False, name=''):
    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv1', trainable=trainable))(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'pool1', trainable=trainable))(x)

    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv2', trainable=trainable))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'pool2', trainable=trainable))(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv3/conv3_1', trainable=trainable))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv3/conv3_2', trainable=trainable))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'pool3', trainable=trainable))(x)

    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv4/conv4_1', trainable=trainable))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name=name+'conv4/conv4_2', trainable=trainable))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'pool4', trainable=trainable))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(trainable=trainable))(x)
    return x
