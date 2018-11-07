
import os
import tensorflow as tf
import dataset_interface as dts
import MSYNC.loss as loss
import MSYNC.stats as stats
from MSYNC.Models import STFTModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logname = 'regression_stft_dnn_3sec_auto'

model_params = {'stft_frame_length': 512,
                'stft_frame_step': 1,
                'input_shape': (20480,),
                'outdim_size': 32,
                'pre_train_lr': 0.0001,
                'dctw_lr': 0.0001,
                'reg_lr': 0.0001,
                'dctw_weights_file': './saved_models/%s_dctw_weights.h5' % logname,
                'reg_weights_file': './saved_models/%s_reg_weights.h5' % logname,
                'num_classes': 1  #2 * 10240 // 20
                }

data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//8,
               'example_length': 20480,
               'batch_size': 4,
               'repeat': 100000,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 20480 // 20
               }


# Get DCTW Model
model = STFTModel(model_params)
model.build_branch_models()
dctw_model = model.build_dctw_model()

# DCTW Training
if not os.path.isfile(model_params['dctw_weights_file']):
    print('Training DCTW...')
    data_params['batch_size'] = 1
    dctw_data = dts.dctw_pipeline(data_params)
    dctw_cp = tf.keras.callbacks.ModelCheckpoint('./logs/%s/dctw/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
    dctw_st = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    dctw_tb = stats.TensorBoardDTW(log_dir='./logs/%s/dctw' % logname, histogram_freq=8, batch_size=data_params['batch_size'], write_images=True)

    dctw_model.summary()
    dctw_model.compile(loss=loss.cca_loss(model_params['outdim_size'], True), optimizer=tf.keras.optimizers.RMSprop(lr=model_params['dctw_lr'], clipnorm=1.0))
    dctw_model.fit(dctw_data, epochs=400, steps_per_epoch=25, validation_data=dctw_data, validation_steps=25, callbacks=[dctw_tb, dctw_cp, dctw_st])
    dctw_model.save_weights(model_params['dctw_weights_file'])


# Freeze DCTW Model and get Regression Model
model.freeze_branch_models()
reg_model = model.build_reg_model()

# Classification Training
print('Training Regressor...')
data_params['batch_size'] = 1
ref_data = dts.regression_pipeline(data_params)
reg_cp = tf.keras.callbacks.ModelCheckpoint('./logs/%s/reg/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
reg_st = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
reg_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/%s/reg' % logname, histogram_freq=8, batch_size=data_params['batch_size'], write_images=True)

reg_model.summary()
reg_model.load_weights(model_params['dctw_weights_file'], by_name=True)
reg_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(lr=model_params['reg_lr'], clipnorm=1.0))
reg_model.fit(ref_data, epochs=400, steps_per_epoch=25, validation_data=ref_data, validation_steps=25, callbacks=[reg_tb, reg_cp, reg_st])
reg_model.save_weights(model_params['reg_weights_file'])
