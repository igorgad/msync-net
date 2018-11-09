
import os
import tensorflow as tf
import dataset_interface as dts
import MSYNC.loss as loss
import MSYNC.stats as stats
from MSYNC.Model import MSYNCModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logname = 'avenet_logmel_autotest'

train_params = {'lr': 0.0001,
                'weights_file': './saved_models/%s_dctw_weights.h5' % logname,
                }

data_params = {'dataset_file': './data/BACH10/MSYNC-bach10.tfrecord',
               'audio_root': './data/BACH10/Audio',
               'sample_rate': 16000,
               'example_length': 15360,  # almost 1 second of audio
               'batch_size': 32,
               'sequential_batch_size': 16,
               'repeat': 100000,
               'shuffle_buffer': 32,
               'scale_value': 1.0,
               'max_delay': 4 * 15360
               }

# Get Model
msync_model = MSYNCModel(input_shape=(15360,))
model = msync_model.build_model()

# Get data pipelines
train_data = dts.train_pipeline(data_params)
val_data = dts.test_pipeline(data_params)

# Classification Training
checkpoint = tf.keras.callbacks.ModelCheckpoint('./logs/%s/model-checkpoint.hdf5' % logname, monitor='val_loss', period=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
tensorboard = stats.TensorBoardAVE(log_dir='./logs/%s' % logname, histogram_freq=4, batch_size=data_params['batch_size'], write_images=True)

model.summary()
model.compile(loss=loss.contrastive_loss, optimizer=tf.keras.optimizers.Adam(lr=train_params['lr']))
model.fit(train_data, epochs=400, steps_per_epoch=25, validation_data=val_data, validation_steps=25, callbacks=[checkpoint, early_stop, tensorboard])
model.save_weights(train_params['weights_file'])
