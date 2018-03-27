import scipy.io
import os
import errno
import shutil
from os.path import isfile, join, isdir


def dir_to_file_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]


def dir_to_file_list_with_ext(path_to_dir, ext):
    return list(filter(lambda x: x.endswith(ext), dir_to_file_list(path_to_dir)))


def dir_to_subdir_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isdir(join(path_to_dir, f))]


def create_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def delete_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def delete_file(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_file_name(path):
    path = path.replace('\\', '/')
    return path.split('/')[-1]


def get_file_name_without_ext(path):
    name_with_ext = get_file_name(path)
    return name_with_ext.split(".")[0]


def read_mat(file):
    return scipy.io.loadmat(file)


def mat_to_array(file, field='signal'):
    mat = read_mat(file)
    return mat[field]


def read_list_from_file(file_path, is_int=False):
    with open(file_path, "r") as file:
        l = []
        for line in file:
            if is_int:
                l.append(int(line.rstrip()))
            else:
                l.append(float(line.rstrip()))
        return l


def write_list_to_file(data_list, file_path):
    with open(file_path, "a") as file:
        for item in data_list:
            file.write(str(item) + "\n")


# if __name__ == '__main__':
    # mat = read_mat('D:\\private_git\\DroneNoiseSuppression\\Data\\Rec3_3_resampled.mat')
    # print(mat['Channel_1_Data_resample'].shape)
    # write_list_to_file(mat['Channel_1_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_1')
    # write_list_to_file(mat['Channel_2_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_2')
    # write_list_to_file(mat['Channel_3_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_3')
    # write_list_to_file(mat['Channel_4_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_4')
    # write_list_to_file(mat['Channel_5_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_5')
    # write_list_to_file(mat['Channel_6_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_6')
    # write_list_to_file(mat['Channel_7_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_7')
    # write_list_to_file(mat['Channel_8_Data_resample'], 'D:\\private_git\\DroneNoiseSuppression\\Data\\Extracted\\Rec3_3_Channel_8')