
import os
import tensorflow as tf
import numpy as np
import dataset_interface as dts
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

iterations = 50
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
               'audio_root': dataset_audio_root,
               'split_seed': 2,
               'split_rate': 0.8
               }

f = open('./logs/min_loss_lr_drop_rand_3fc128feat.txt', 'w')

#####################################################################################
lrs = np.random.uniform(1e-6, 1e-3, iterations)
loss = []

for lr in lrs:
    tf.set_random_seed(0)
#     data_params['split_seed'] = rand_min_loss
    train_data, _ = dts.pipeline(data_params)
    msync_model = MSYNCModel(input_shape=(data_params['sequential_batch_size'], data_params['example_length']))
    model = msync_model.build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=lr))
    hist = model.fit(train_data, epochs=4, steps_per_epoch=25)
    loss.append(hist.history['loss'][-1])

    print('**********************************************')
    print ('LR FINDER:' + str(len(loss)) + ' - lr: ' + str(lr) + ', loss: ' + str(np.min(hist.history['loss'])) + '. min_loss: ' + str(np.min(np.array(loss))))
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
