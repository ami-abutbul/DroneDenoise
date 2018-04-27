import os
import platform

#########################################################
# speech data sets
#########################################################
TSPspeech_path = '/home/amimichael/DroneDenoise/Data/TSPspeech/'


#########################################################
# parameters
#########################################################
number_of_background_noise = 40

record_per_mat_data = 10
batch_size = 512
time_series = 200
window_size = 960
epochs_per_mat = 10
iterations = 100
mat_data_width = window_size + (time_series - 1)*(window_size // 2)

mode = 'train' #test
loss_file = 'loss.log'
model_dir = 'models_weights/model1'
model_saved_weights = model_dir + '/model.h5'
cuda_visible_devices = '1'
if platform.system() == 'Linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices