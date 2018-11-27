
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import MSYNC.utils as utils
import MSYNC.stats as stats
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_params = {'lr': 1e-4,
                'drop_lr': 0.1,
                'drop_epoch': 5,
                'pretrain': False
                }

dataset = 'bach10'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,
               'sequential_batch_size': 8,
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 2,
               'split_rate': 0.7,
               'debug_auto': False
               }

logname = 'master-' + dataset + ''.join(['-%s=%s' % (key, value) for (key, value) in train_params.items()])
logname = logname + ''.join(['-%s=%s' % (key, str(value).replace(' ', '_')) for (key, value) in data_params.items()])
print (logname)

# Get data pipelines
data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root
train_data, validation_data = dts.pipeline(data_params)

# Set Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('./logs/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir='./logs/%s' % logname, histogram_freq=4, batch_size=data_params['random_batch_size'], write_images=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: train_params['lr'] * np.power(train_params['drop_lr'], np.floor((1 + epoch) / train_params['drop_epoch'])))
callbacks = [checkpoint, tensorboard, lr_scheduler, early_stop]

# Get Model
msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']), use_pretrain=train_params['pretrain'])
model = msync_model.build_model()
model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=train_params['lr']), metrics=['accuracy', utils.range_categorical_accuracy])
model.fit(train_data, epochs=400, steps_per_epoch=25, validation_data=validation_data, validation_steps=25, callbacks=callbacks)
