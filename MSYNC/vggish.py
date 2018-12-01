
import tensorflow as tf

# weight path
WEIGHTS_PATH = './saved_models/vggish_audioset_weights_without_fc2.h5'


def vggish(input, trainable=True, name=''):
    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block1/conv')(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'block1/bn')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block1/elu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'block1/pool')(x)
    
    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block2/conv')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'block2/bn')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block2/elu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'block2/pool')(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block3/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'block3/bn1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block3/elu1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block3/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'block3/bn2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block3/elu2')(x)    
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'block3/pool')(x)
    
    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block4/conv1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'block4/bn1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block4/elu1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False), trainable=trainable, name=name+'block4/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name + 'block4/bn2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name+'block4/elu2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'block4/pool')(x)

    # Global Average
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), trainable=trainable, name=name+'GAverage')(x)
    return x
