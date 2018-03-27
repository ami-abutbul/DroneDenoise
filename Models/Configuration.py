import os
import platform

#########################################################
# speech data sets
#########################################################
TSPspeech_path = 'C:/Users/il115552/Desktop/TSPspeech'


#########################################################
# parameters
#########################################################
number_of_background_noise = 20


cuda_visible_devices = '1'
if platform.system() == 'Linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices