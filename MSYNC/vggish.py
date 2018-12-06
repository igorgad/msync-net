
import tensorflow as tf

# weight path
WEIGHTS_PATH = './saved_models/vggish_audioset_weights_without_fc2.h5'


def vggish(input, trainable=True, name=''):
    # Block 1
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block1/conv1')(input)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block1/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block1/bn')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block1/elu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_block1/pool')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block1/bn')(x)
    
    

    # Block 2
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block2/conv1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block2/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block2/bn')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block2/elu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_block2/pool')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block2/bn')(x)

    # Block 3
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block3/conv1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block3/bn1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block3/elu1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block3/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block3/bn2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block3/elu2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_block3/pool')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block3/bn2')(x)
    

    # Block 4
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block4/conv1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block4/bn1')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block4/elu1')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='linear', padding='same', use_bias=False), trainable=trainable, name=name+'vgg_block4/conv2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block4/bn2')(x)
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.ELU(), name=name + 'vgg_block4/elu2')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'), trainable=trainable, name=name+'vgg_block4/pool')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), trainable=trainable, name=name+'vgg_block4/bn2')(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), trainable=trainable, name=name+'GAverage')(x)
    return x
