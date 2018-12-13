
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
import MSYNC.utils as utils
import MSYNC.stats as stats
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.set_random_seed(0)

model_params = {'lr': 6.3e-5,
                'stft_window': 1600,
                'stft_step': 160,
                'num_mel_bins': 128,
                'num_spectrogram_bins': 1025,
                'sample_rate': 16000,
                'lower_edge_hertz': 125.0,
                'upper_edge_hertz': 7500.0,
                'lstm_units': [128, 256, 512],
                'top_units': [256, 128],
                'dropout': 0.5,
                'optimizer': 'adam'
                }

dataset = 'medleydb_v2'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB_v2.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 4 * 15360,  # almost 1 second of audio
               'random_batch_size': 16,
               'sequential_batch_size': 8,
               'max_delay': 16 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar'
               }


logname = 'no_ae-lstm/' + dataset + '/'
logname = logname + ''.join(['%s=%s-' % (key, str(value).replace(' ', '_')) for (key, value) in data_params.items()]) + '/'
logname = logname + ''.join(['%s=%s-' % (key, str(value).replace(' ', '')) for (key, value) in model_params.items()])
print (logname)

# Get data pipelines
data_params['split_seed'] = 3
data_params['split_rate'] = 0.8
data_params['debug_auto'] = False
data_params['scale_value'] = 1.0
data_params['shuffle_buffer'] = 32
data_params['dataset_file'] = dataset_file
data_params['audio_root'] = dataset_audio_root
train_data, validation_data = dts.pipeline(data_params)

# Set callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('./logs/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir='./logs/%s' % logname, histogram_freq=0, batch_size=data_params['random_batch_size'], write_images=True)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpoint, tensorboard, lr_reducer]

# Get Model
msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']), model_params=model_params)
model = msync_model.build_model()
model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=model_params['lr']), metrics=['accuracy', utils.range_categorical_accuracy])
model.fit(train_data, epochs=50, steps_per_epoch=25, validation_data=validation_data, validation_steps=25, callbacks=callbacks)
print (logname)
