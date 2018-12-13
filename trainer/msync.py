
import os
import tensorflow as tf
import argparse
import trainer.dataset_interface as dataset_interface
import trainer.utils as utils
import trainer.stats as stats
from trainer.Model import MSYNCModel

tf.set_random_seed(26)

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 128,
               'sequential_batch_size': 8,
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0
               }

model_params = {'stft_window': 1600,
                'stft_step': 160,
                'num_mel_bins': 128,
                'num_spectrogram_bins': 1025,
                'lower_edge_hertz': 125.0,
                'upper_edge_hertz': 7500.0,
                'encoder_arch': 'lstm',
                'encoder_units': [64, 128, 256],
                'top_units': [128, 8],
                'dropout': 0.25
                }

train_params = {'lr': 6.3e-5,
                'epochs': 150,
                'steps_per_epoch': 25,
                'val_steps': 25
                }

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
parser.add_argument('--logdir', type=str, default='./logs/', help='The directory to store the experiments logs (default: ./logs/)')
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--dataset_audio_dir', type=str, default=dataset_audio_root, help='Directory to fetch wav files (default: %s)' % str(dataset_audio_root))
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in train_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]

params = parser.parse_known_args()[0]
logname = 'master/' + ''.join(['%s=%s/' % (key, str(val).replace('/', '_').replace(' ', '')) for key, val in sorted(list(params.__dict__.items()))]) + 'run'

# Set callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(params.logdir + '/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir=params.logdir + '/%s' % logname, histogram_freq=4, batch_size=params.random_batch_size, write_images=True)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpoint, tensorboard, lr_reducer]

# Build Data Pipeline and Model
train_data, validation_data = dataset_interface.pipeline(params)
msync_model = MSYNCModel(input_shape=(params.sequential_batch_size, params.example_length), model_params=params)
model = msync_model.build_model()
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=params.lr), metrics=['accuracy', utils.range_categorical_accuracy])
model.fit(train_data, epochs=params.epochs, steps_per_epoch=params.steps_per_epoch, validation_data=validation_data, validation_steps=params.val_steps, callbacks=callbacks)
print (logname)
