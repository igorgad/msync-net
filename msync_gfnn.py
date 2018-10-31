
import os
import tensorflow as tf
import dataset_interface as dts
import MSYNC.gfnn_model as gfnn_model
import MSYNC.loss as loss
import MSYNC.stats as stats
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
importlib.reload(dts)
importlib.reload(gfnn_model)
importlib.reload(stats)

logname = 'gfnn_lstm_classification_r0'

osc_params = {'f_min': 200.0,
              'f_max': 5000.0,
              'alpha': -1.0,
              'beta1': -10.0,
              'beta2': 0.0,
              'delta1': 0.0,
              'delta2': 0.0,
              'eps': 1.0,
              'k': 1.0
              }

model_params = {'num_osc': 256,
                'dt': 1/(44100//4),
                'osc_params': osc_params,
                'input_shape': (10240,),
                'outdim_size': 128,
                'pre_train_lr': 0.0001,
                'dctw_lr': 0.0001,
                'class_lr': 0.0001,
                'v1_weights_file': './saved_models/v1_%s_weights.h5' % logname,
                'v2_weights_file': './saved_models/v2_%s_weights.h5' % logname,
                'dctw_weights_file': './saved_models/dctw_%s_weights.h5' % logname,
                'class_weights_file': './saved_models/class_%s_weights.h5' % logname,
                'num_classes': 1  # 2 * 10240 // 20
                }

data_params = {'dataset_file': './data/BACH10/msync-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 44100//4,
               'example_length': 10240,
               'batch_size': 8,
               'repeat': 10000,
               'shuffle_buffer': 32,
               'scale_value': 0.2,
               'max_delay': 10240 // 20
               }

# Get models
class_model, dctw_model, v1_model, v2_model = gfnn_model.build_models(model_params)

# Apply denoising autoencoder pre-training if necessary
if not os.path.isfile(model_params['v1_weights_file']):
    print ('Pre-training branch 1')
    v1_model.summary()
    v1_data = dts.v1_pipeline(data_params)
    v1_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/%s/pre-train_v1' % logname)
    v1_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=1, mode='auto')
    v1_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['pre_train_lr']))
    v1_model.fit(v1_data, epochs=40, steps_per_epoch=10, callbacks=[v1_tb, v1_st])
    v1_model.save_weights(model_params['v1_weights_file'])
    del v1_model

if not os.path.isfile(model_params['v2_weights_file']):
    print('Pre-training branch 2')
    v2_model.summary()
    v2_data = dts.v2_pipeline(data_params)
    v2_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/%s/pre-train_v2' % logname)
    v2_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=1, mode='auto')
    v2_model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(lr=model_params['pre_train_lr']))
    v2_model.fit(v2_data, epochs=40, steps_per_epoch=10, callbacks=[v2_tb, v2_st])
    v2_model.save_weights(model_params['v2_weights_file'])
    del v2_model

# DCTW Training
if not os.path.isfile(model_params['dctw_weights_file']):
    print('Training DCTW...')
    data_params['batch_size'] = 1
    dctw_data = dts.dctw_pipeline(data_params)
    dctw_cp = tf.keras.callbacks.ModelCheckpoint('./logs/%s/dctw0/model-checkpoint.hdf5' % logname, monitor='val_loss', period=4, save_best_only=True)
    dctw_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=1, mode='auto')
    dctw_tb = stats.TensorBoardDTW(log_dir='./logs/%s/dctw0' % logname, histogram_freq=4, batch_size=data_params['batch_size'], write_images=True)

    dctw_model.summary()
    dctw_model.load_weights(model_params['v1_weights_file'], by_name=True)
    dctw_model.load_weights(model_params['v2_weights_file'], by_name=True)
    dctw_model.compile(loss=loss.cca_loss(model_params['outdim_size'], False), optimizer=tf.keras.optimizers.RMSprop(lr=model_params['dctw_lr'], clipnorm=1.0))
    dctw_model.fit(dctw_data, epochs=400, steps_per_epoch=20, validation_data=dctw_data, validation_steps=10, callbacks=[dctw_tb, dctw_cp, dctw_st])
    dctw_model.save_weights(model_params['dctw_weights_file'])

# Classification Training
print('Training Classifier...')
data_params['batch_size'] = 2
class_data = dts.softmax_pipeline(data_params)
class_cp = tf.keras.callbacks.ModelCheckpoint('./logs/%s/class0/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
class_st = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=1, mode='auto')
class_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/%s/class0' % logname, histogram_freq=4, batch_size=data_params['batch_size'], write_images=True)

class_model.summary()
class_model.load_weights(model_params['dctw_weights_file'], by_name=True)
class_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=model_params['class_lr'], clipnorm=1.0), metrics=['accuracy'])
class_model.fit(class_data, epochs=400, steps_per_epoch=20, validation_data=class_data, validation_steps=10, callbacks=[class_tb, class_cp, class_st])
class_model.save_weights(model_params['class_weights_file'])
