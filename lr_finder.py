
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = 'medleydb'
dataset_file = './data/BACH10/MSYNC-bach10.tfrecord' if dataset == 'bach10' else './data/MedleyDB/MSYNC-MedleyDB.tfrecord'
dataset_audio_root = './data/BACH10/Audio' if dataset == 'bach10' else './data/MedleyDB/Audio'

data_params = {'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'random_batch_size': 16,  # For training
               'sequential_batch_size': 8,  # For validation
               'max_delay': 4 * 15360,
               'instrument_1': 'bassoon' if dataset == 'bach10' else 'electric bass',         # Only valid for MedleyDB dataset
               'instrument_2': 'clarinet' if dataset == 'bach10' else 'clean electric guitar',  # Only valid for MedleyDB dataset
               'debug_auto': False,
               'scale_value': 1.0,
               'shuffle_buffer': 1,
               'dataset_file': dataset_file,
               'audio_root': dataset_audio_root
               }

f = open('./logs/min_loss_lr_newdtsi.txt', 'w')
lrs = np.random.uniform(1e-6, 1e-4, 100)
loss = []

for lr in lrs:
    train_data, _ = dts.pipeline(data_params)
    msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']), use_pretrain=False)
    model = msync_model.build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr))
    hist = model.fit(train_data, epochs=4, steps_per_epoch=25)
    loss.append(np.min(hist.history['loss']))

    print('**********************************************')
    print (str(len(loss)) + ' - lr: ' + str(lr) + ', loss: ' + str(np.min(hist.history['loss'])) + '. min_loss: ' + str(np.min(np.array(loss))))
    print('**********************************************')

    tf.keras.backend.clear_session()

# Get minimum loss and better lr
loss = np.array(loss)
min_loss = np.min(loss)
lr_min_loss = lrs[np.argmin(loss)]

print('**********************************************')
print('**********************************************')
print ('COARSE min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss))
print('**********************************************')

f.write('COARSE min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss) + '\n')
lrs = np.random.uniform(lr_min_loss - 1e-5, lr_min_loss + 1e-5, 100)
loss = []

for lr in lrs:
    train_data, _ = dts.pipeline(data_params)
    msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']), use_pretrain=False)
    model = msync_model.build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr))
    hist = model.fit(train_data, epochs=4, steps_per_epoch=25)
    loss.append(np.min(hist.history['loss']))

    print('**********************************************')
    print (str(len(loss)) + ' - lr: ' + str(lr) + ', loss: ' + str(np.min(hist.history['loss'])) + '. min_loss: ' + str(np.min(np.array(loss))))
    print('**********************************************')

    tf.keras.backend.clear_session()

# Get minimum loss and better lr
loss = np.array(loss)
min_loss = np.min(loss)
lr_min_loss = lrs[np.argmin(loss)]

print('**********************************************')
print('**********************************************')
print ('FINE min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss))
print('**********************************************')

f.write('FINE min_loss of ' + str(min_loss) + ' with lr: ' + str(lr_min_loss) + '\n')
f.close()
