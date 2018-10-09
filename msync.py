
import dataset_interface as dts
import models.simple_gfnn as gfnn_model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import importlib
importlib.reload(dts)
importlib.reload(gfnn_model)


data_params = {'dataset_file': '/media/igor/DATA/Dataset/BACH10/msync-bach10.tfrecord',
               'audio_root': '/media/igor/DATA/Dataset/BACH10/Audio',
               'sample_rate': 44100//4,
               'frame_length': 2048,
               'frame_step': 1024,
               'batch_size': 1
               }

model_params = {'num_osc': 180,
                'dt': 1/(44100//4),
                'input_shape': (2048,),
                'outdim_size': 128,
                'lr': 0.01
                }


data = dts.pipeline(data_params)
model = gfnn_model.simple_gfnn_cca_v0(model_params)

model.fit(data, epochs=10, steps_per_epoch=30)
