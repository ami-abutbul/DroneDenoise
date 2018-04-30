from DroneDenoise.Models.Configuration import *
from DroneDenoise.Models.Workers import create_workers
from DroneDenoise.DataHandler.DataHandler import SignalsHandler
from DroneDenoise.Utilities.file_utils import create_dir, write_list_to_file
from DroneDenoise.Models.model1 import Model1
from DroneDenoise.Models.model2 import Model2
from DroneDenoise.Models.model3 import Model3
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import multiprocessing
import tensorflow as tf
import scipy.io.wavfile
import numpy as np


def get_model():
    return Model3().build_model()


def train(signals_dir):
    model = get_model()
    create_dir(model_dir)

    if load_weights:
        model.load_weights(model_saved_weights)

    data_queue = multiprocessing.Manager().Queue(5)
    processes_list = create_workers(num_of_workers, data_queue, signals_dir, do_augmentation)

    checkpointer = ModelCheckpoint(filepath=model_saved_weights, monitor='loss', verbose=2, save_best_only=True, mode='min')
    losses = []
    current_lr = initial_lr
    for i in range(iterations):
        print("-" * 50)
        print('iteration: {}'.format(i))
        print("-" * 50)

        if i != 0 and i % 5 == 0:
            current_lr = current_lr * 0.9

        def lr_decay(epoch):
            return current_lr

        x, y = data_queue.get()
        hist = model.fit(x, y,  validation_split=0.1, epochs=epochs_per_mat, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpointer, LearningRateScheduler(lr_decay)])
        losses.append(hist.history['loss'])
    write_list_to_file(losses, loss_file)

    for p in processes_list:
        p.terminate()
    print('Done training')


def test():
    model = get_model()
    model.load_weights(model_saved_weights)
    Fs, wav_data = scipy.io.wavfile.read('../DataHandler/noise.wav')
    test_signal = SignalsHandler.create_test(wav_data)
    test_signal = test_signal[:,:5900,:]
    test_signal = np.reshape(test_signal, [-1, 100, 480])
    # prediction = model.predict(test_signal)
    prediction = test_signal
    prediction = np.reshape(prediction, [1, -1, 480])
    predicted_signal = SignalsHandler.reconstruct_windowed_signal(prediction)
    scipy.io.wavfile.write("./test_predicted.wav", Fs, np.array(predicted_signal, dtype='float64'))
    scipy.io.wavfile.write("./clean_predicted.wav", Fs, np.array(predicted_signal - wav_data[:len(predicted_signal)], dtype='float64'))


def catch_gpu():
    x = tf.constant(2)
    y = tf.constant(3)
    z = x + y
    with tf.Session() as sess:
        sess.run(z)


if __name__ == '__main__':
    catch_gpu()
    if mode == "train":
        print("Start training ..")
        if platform.system() == 'Linux':
            train('/home/amimichael/DroneDenoise/Data/Extracted_Raw_Drone/sides/')
        else:
            train('D:\\private_git\\DroneDenoise\\Data\\Extracted Raw Drone')

    if mode == "test":
        print("Start prediction ..")
        if platform.system() == 'Linux':
            test()
        else:
            test()