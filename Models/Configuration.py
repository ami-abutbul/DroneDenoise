import os
import platform

#########################################################
# speech data sets
#########################################################
if platform.system() == 'Linux':
    TSPspeech_path = '/home/amimichael/DroneDenoise/Data/TSPspeech/'
else:
    TSPspeech_path = 'D:\\private_git\\DroneDenoise\\Data\\TSPspeech'


#########################################################
# parameters
#########################################################
number_of_background_noise = 40
num_of_workers = 7

record_per_mat_data = 20
batch_size = 256
time_series = 100
window_size = 960
epochs_per_mat = 4
iterations = 500
mat_data_width = window_size + (time_series - 1)*(window_size // 2)
take_half_window = False
if take_half_window:
    window_size = window_size // 2

mode = 'test' #test
load_weights = False
do_augmentation = False
initial_lr = 1e-3

loss_file = 'loss.log'
model_dir = 'models_weights/model2'
model_saved_weights = model_dir + '/model.h5'

cuda_visible_devices = '1'
if platform.system() == 'Linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices