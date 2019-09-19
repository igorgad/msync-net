
import os
import tensorflow as tf
import argparse
import trainer.dataset_interface as dataset_interface
import trainer.utils as utils
import trainer.stats as stats
from trainer.Model import MSYNCModel
from tensorflow.python.lib.io import file_io

tf.set_random_seed(26)

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB_v2.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

sample_rate = 44100
block_size = 1024

data_params = {'sample_rate': sample_rate,
               'example_length': int(4 * sample_rate / block_size),
               'num_examples': 1,
               'max_delay': int(2 * sample_rate / block_size),
               'labels_precision': sample_rate // 1,
               'random_batch_size': 16,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'drum set',  #'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'electric bass',  #'clean electric guitar',
               'type_1': 'brass',
               'type_2': 'strings',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 1000,
               'from_bucket': False,
               'block_size': block_size
               }

model_params = {'encoder_type': 'cnn',
                'encoder_units': [128, 256, 512],
                'top_units': [128],
                'post_ecl_units': [],
                'post_ecl_pooling': False,
                'ecl_end_strategy': 'diag_mean', #or diag_mean
                'dropout': 0.5,
                'dmrn': False,
                'residual_connection': False,
                'rnn_cell': 'LSTM',
                'culstm': True,
                'bw': 1.0
                }

train_params = {'lr': 1.0e-4,
                'epochs': 50,
                'steps_per_epoch': 50,
                'val_steps': 50,
                'metrics_range': [sample_rate // 1 // block_size, 
                                  sample_rate // 2 // block_size, 
                                  sample_rate // 4 // block_size],
                'verbose': 1,
                'num_folds': 5
                }

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
parser.add_argument('--logdir', type=str, default='logs/', help='The directory to store the experiments logs (default: logs/)')
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--dataset_audio_dir', type=str, default=dataset_audio_root, help='Directory to fetch wav files (default: %s)' % str(dataset_audio_root))
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in train_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]

params = parser.parse_known_args()[0]
logname = '3master-VBR/single-fold/cnn-allrelu-v2/bw1/llr-plateau05-patience4/typefilt-rndtype/' + ''.join(['%s=%s/' % (key, str(val).replace('/', '').replace(' ', '').replace('gs:', '')) for key, val in sorted(list(params.__dict__.items()))]) + 'run'

if params.logdir.startswith('gs://'):
    os.system('mkdir -p %s' % logname)
    checkpoint_file = logname + '/model-checkpoint.hdf5'
else:
    checkpoint_file = os.path.join(params.logdir, logname + '/model-checkpoint.hdf5')

# Set callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir=os.path.join(params.logdir, logname), histogram_freq=4, batch_size=params.random_batch_size, write_images=True, range=params.metrics_range[0])
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpoint, tensorboard, lr_reducer]
metrics = [utils.topn_range_categorical_accuracy(n=n, range=range) for n in [1, 5] for range in params.metrics_range]
loss = tf.keras.losses.binary_crossentropy

# Build Data Pipeline and Model
train_data, validation_data = dataset_interface.pipeline(params)
msync_model = MSYNCModel(model_params=params)
model = msync_model.build_model()
model.summary()
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=params.lr), metrics=metrics)
model.fit(train_data, epochs=params.epochs, steps_per_epoch=params.steps_per_epoch, validation_data=validation_data, validation_steps=params.val_steps, callbacks=callbacks, verbose=params.verbose)


if params.logdir.startswith('gs://'):
    print('transferring model checkpoint hdf5 to bucket...')
    with file_io.FileIO(checkpoint_file, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(params.logdir, logname + '/model-checkpoint.hdf5'), mode='w+') as output_f:
            output_f.write(input_f.read())
