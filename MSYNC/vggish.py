
import tensorflow as tf

# weight path
WEIGHTS_PATH = './saved_models/vggish_audioset_weights_without_fc2.h5'


def vggish(input, trainable=False, name=''):
    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv1')(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'pool1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vbn1')(x)
    

    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'pool2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'vbn2')(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv3/conv3_1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv3/conv3_2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'pool3')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'vbn3')(x)
    

    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv4/conv4_1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='elu', padding='same', use_bias=False), trainable=trainable, name=name+'conv4/conv4_2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'pool4')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'vbn4')(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), trainable=trainable, name=name+'GAverage')(x)
    return x
