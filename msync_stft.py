
import os
import tensorflow as tf
import dataset_interface as dts
import MSYNC.stft_model as stft_model
import MSYNC.loss as loss
import MSYNC.stats as stats
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
importlib.reload(dts)
importlib.reload(stft_model)
importlib.reload(stats)

model_params = {'stft_frame_length': 512,
                'stft_frame_step': 256,
                'input_shape': (8192,),
                'outdim_size': 128,
                'pre_train_lr': 0.001,
                'dctw_lr': 0.01,
                'v1_weights_file': './models/v1_stft_weights.h5',
                'v2_weights_file': './models/v2_stft_weights.h5',
                }

data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'example_length': 8192,
               'batch_size': 1,
               'repeat': 40,
               'shuffle_buffer': 32,
               'scale_value': 1.0
               }

# Get models
dctw_model, v1_model, v2_model = stft_model.simple_stft_cca(model_params)

# Apply denoising autoencoder pre-training if necessary
if not os.path.isfile(model_params['v1_weights_file']):
    print ('Pre-training branch 1')
    v1_data = dts.v1_pipeline(data_params)
    v1_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/v0/pre-train_v1')
    v1_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=1, mode='auto')
    v1_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['pre_train_lr']))
    v1_model.fit(v1_data, epochs=20, steps_per_epoch=8, callbacks=[v1_tb, v1_st])
    v1_model.save_weights(model_params['v1_weights_file'])
    del v1_model

if not os.path.isfile(model_params['v2_weights_file']):
    print('Pre-training branch 2')
    v2_data = dts.v2_pipeline(data_params)
    v2_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/v0/pre-train_v2')
    v2_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=1, mode='auto')
    v2_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['pre_train_lr']))
    v2_model.fit(v2_data, epochs=20, steps_per_epoch=8, callbacks=[v2_tb, v2_st])
    v2_model.save_weights(model_params['v2_weights_file'])
    del v2_model

# DCTW Training
print ('Training DCTW...')
dctw_data = dts.dctw_pipeline(data_params)
dctw_tb = stats.TensorBoardDTW(log_dir='./logs/v0/dctw', histogram_freq=1, batch_size=data_params['batch_size'], write_images=True)

dctw_model.load_weights(model_params['v1_weights_file'], by_name=True)
dctw_model.load_weights(model_params['v2_weights_file'], by_name=True)
dctw_model.compile(loss=loss.cca_loss, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['dctw_lr']))
dctw_model.fit(dctw_data, epochs=20, steps_per_epoch=8, validation_data=dctw_data, validation_steps=1, callbacks=[dctw_tb])
