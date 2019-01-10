
import os
import tensorflow as tf
import numpy as np
import trainer.dataset_interface as dts
from trainer.Model import MSYNCModel
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

iterations = 50
dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 4 * 15360,
               'max_delay': 2 * 15360,
               'labels_precision': 15360 // 1,
               'random_batch_size': 16,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',
               'split_seed': 3,
               'split_rate': 0.8,
               'debug_auto': False,
               'scale_value': 1.0,
               'limit_size_seconds': 1000,
               'from_bucket': False
               }

model_params = {'stft_window': 3200,
                'stft_step': 160,
                'num_mel_bins': 256,
                'num_spectrogram_bins': 2049,
                'lower_edge_hertz': 125.0,
                'upper_edge_hertz': 7500.0,
                'encoder_arch': 'lstm',
                'encoder_units': [512, 256],
                'top_units': [256, 128],
                'dropout': 0.5,
                'dmrn': False,
                'residual_connection': False,
                'culstm': True
                }

train_params = {'lr': 1.0e-4,
                'epochs': 60,
                'steps_per_epoch': 25,
                'val_steps': 25
                }

parser = argparse.ArgumentParser(description='Launch training session of msync-net.')
parser.add_argument('--logdir', type=str, default='logs/', help='The directory to store the experiments logs (default: logs/)')
parser.add_argument('--dataset_file', type=str, default=dataset_file, help='Dataset file in tfrecord format (default: %s)' % str(dataset_file))
parser.add_argument('--dataset_audio_dir', type=str, default=dataset_audio_root, help='Directory to fetch wav files (default: %s)' % str(dataset_audio_root))
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in train_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in model_params.items()]
[parser.add_argument('--%s' % key, type=type(val), help='%s' % val, default=val) for key, val in data_params.items()]

params = parser.parse_known_args()[0]

f = open('./logs/min_loss_lr_drop_rand_3fc128feat.txt', 'w')

#####################################################################################
lrs = np.random.uniform(1e-6, 1e-3, iterations)
loss = []

for lr in lrs:
    print('**********************************************')
    print('RUNNING WITH LR: ' + str(lr))
    print('**********************************************')

    tf.set_random_seed(26)
#     data_params['split_seed'] = rand_min_loss
    train_data, validation_data = dts.pipeline(params)
    msync_model = MSYNCModel(input_shape=(params.example_length,), model_params=params)
    model = msync_model.build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr))
    hist = model.fit(train_data, epochs=4, steps_per_epoch=25, validation_data=validation_data, validation_steps=10)
    loss.append(hist.history['val_loss'][-1])

    print('**********************************************')
    print ('LR FINDER:' + str(len(loss)) + ' - lr: ' + str(lr) + ', loss: ' + str(np.min(hist.history['val_loss'])) + '. min_loss: ' + str(np.min(np.array(loss))))
    print('**********************************************')

    tf.keras.backend.clear_session()

# Get minimum loss and better lr
loss = np.array(loss)
min_loss = np.min(loss)
lr_min_loss = lrs[np.argmin(loss)]

print('**********************************************')
print('**********************************************')
print ('LR FINDER min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss))
print('**********************************************')
print('**********************************************')
f.write('LR FINDER min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss) + '\n')

#####################################################################################
# drops = np.random.uniform(0.2, 0.8, iterations)
# drop_loss = []

# for drop in drops:
#     tf.set_random_seed(0)
# #     data_params['split_seed'] = rand_min_loss
#     train_data, validation_data = dts.pipeline(data_params)
#     msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']))
#     model = msync_model.build_model()
#     model.dropout_rate = drop
#     model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr_min_loss))
#     hist = model.fit(train_data, epochs=4, steps_per_epoch=150, validation_data=validation_data, validation_steps=25)
#     drop_loss.append(hist.history['val_loss'][-1])

#     print('**********************************************')
#     print ('DROP FINDER:' + str(len(drop_loss)) + ' - dropout: ' + str(drop) + ', loss: ' + str(np.min(hist.history['val_loss'])) + '. min_loss: ' + str(np.min(np.array(drop_loss))))
#     print('**********************************************')

#     tf.keras.backend.clear_session()

# # Get minimum loss and better dropout rate
# loss = np.array(drop_loss)
# min_loss = np.min(drop_loss)
# drop_min_loss = drops[np.argmin(drop_loss)]

# print('**********************************************')
# print('**********************************************')
# print ('DROP FINDER min_loss of ' + str(min_loss) + ' with dropout: ' + str(drop_min_loss))
# print('**********************************************')
# print('**********************************************')
# f.write('DROP FINDER min_loss of ' + str(min_loss) + ' with dropout: ' + str(drop_min_loss) + '\n')

# #####################################################################################
# # Get minimum loss and better dropout rate
# rands = np.int32(np.random.uniform(0, 100, iterations))
# rand_loss = []

# for rand in rands:
#     tf.set_random_seed(rand)
# #     data_params['split_seed'] = rand
#     train_data, validation_data = dts.pipeline(data_params)
#     msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']))
#     model = msync_model.build_model()
#     model.dropout_rate = drop_min_loss
#     model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr_min_loss))
#     hist = model.fit(train_data, epochs=4, steps_per_epoch=150, validation_data=validation_data, validation_steps=25)
#     rand_loss.append(hist.history['val_loss'][-1])

#     print('**********************************************')
#     print ('RAND FINDER:' + str(len(rand_loss)) + ' - rand: ' + str(rand) + ', loss: ' + str(np.min(hist.history['val_loss'])) + '. min_loss: ' + str(np.min(np.array(rand_loss))))
#     print('**********************************************')

#     tf.keras.backend.clear_session()

# loss = np.array(rand_loss)
# min_loss = np.min(rand_loss)
# rand_min_loss = rands[np.argmin(rand_loss)]

# print('**********************************************')
# print('**********************************************')
# print ('RAND FINDER min_loss of ' + str(min_loss) + ' with rand: ' + str(rand_min_loss))
# print('**********************************************')
# print('**********************************************')
# f.write('RAND FINDER min_loss of ' + str(min_loss) + ' with rand: ' + str(rand_min_loss) + '\n')

f.close()
