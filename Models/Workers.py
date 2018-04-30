from DroneDenoise.DataHandler.DataHandler import SignalsHandler
from multiprocessing import Process
import numpy as np


def create_workers(workers_amount, data_queue, signals_dir, do_data_augmentation):
    process_list = []
    for i in range(workers_amount):
        print("create worker {}".format(i))
        signal_handler = SignalsHandler(signals_dir, do_augmentation=do_data_augmentation)
        p = Process(target=load_data_mat, args=(signal_handler, data_queue))
        p.start()
        process_list.append(p)
    print('create_workers - Done')
    return process_list


def load_data_mat(signal_handler, data_queue):
    np.random.seed()
    while True:
        x, y = signal_handler.get_dataset()
        data_queue.put((x, y))
