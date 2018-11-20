
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import MSYNC.loss as loss
import MSYNC.stats as stats
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_params = {'lr': 0.0001,
                'drop_lr': 0.1,
                'drop_epoch': 5,
                'pretrain': False
                }

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 128,  # For training
               'sequential_batch_size': 8,  # For validation
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',         # Only valid for MedleyDB dataset
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',  # Only valid for MedleyDB dataset
               'debug_auto': True
               }

logname = 'cmm-' + dataset + ''.join(['-%s=%s' % (key, value) for (key, value) in train_params.items()])
logname = logname + ''.join(['-%s=%s' % (key, str(value).replace(' ', '_')) for (key, value) in data_params.items()])

# Get Model
msync_model = MSYNCModel(input_shape=(data_params['example_length'],), use_pretrain=train_params['pretrain'])
model = msync_model.build_model()

# Get data pipelines
data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root
train_data, validation_data = dts.bach10_pipeline(data_params) if dataset == 'bach10' else dts.medleydb_pipeline(data_params)

# Classification Training
checkpoint = tf.keras.callbacks.ModelCheckpoint('./logs/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir='./logs/%s' % logname, histogram_freq=4, batch_size=data_params['random_batch_size'], write_images=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: train_params['lr'] * np.power(train_params['drop_lr'], np.floor((1 + epoch) / train_params['drop_epoch'])))
callbacks = [checkpoint, tensorboard, early_stop, lr_scheduler]

model.summary()
model.compile(loss=loss.contrastive_loss, optimizer=tf.keras.optimizers.Adam(lr=train_params['lr']), metrics=[loss.min_ecl_distance_accuracy])
model.fit(train_data, epochs=400, steps_per_epoch=50, validation_data=validation_data, validation_steps=50, callbacks=callbacks)
