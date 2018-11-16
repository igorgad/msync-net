
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import MSYNC.utils as utils
import MSYNC.stats as stats
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logname = 'mwaynet_logmel_difftest'

train_params = {'lr': 0.001,
                'weights_file': './saved_models/%s_dctw_weights.h5' % logname,
                }

data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,
               'sequential_batch_size': 8,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 4 * 15360
               }

# Get Model
msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']))
model = msync_model.build_model()

# Get data pipelines
train_data = dts.bach10_pipeline(data_params)

# Classification Training
checkpoint = tf.keras.callbacks.ModelCheckpoint('./logs/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir='./logs/%s' % logname, histogram_freq=2, batch_size=data_params['random_batch_size'], write_images=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: train_params['lr'] * np.power(0.1, np.floor((1 + epoch) / 5))) # Drop = 0.1, epoch_drop = 5
callbacks = [checkpoint, tensorboard, early_stop, lr_scheduler]

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=train_params['lr']), metrics=['accuracy', utils.range_categorical_accuracy])
model.fit(train_data, epochs=400, steps_per_epoch=25, validation_data=train_data, validation_steps=25, callbacks=callbacks)
model.save_weights(train_params['weights_file'])
