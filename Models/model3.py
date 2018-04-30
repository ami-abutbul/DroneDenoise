from DroneDenoise.Models.Configuration import *
from keras.layers import Dense, Input, multiply, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers


class Model3(object):

    def build_model(self):
        print('Build model...')
        main_input = Input(shape=(None, window_size), name='main_input')
        out = Dense(window_size, activation='relu', name='encoder_dense1')(main_input)
        out = Dense(window_size, activation='relu', name='encoder_dense2')(out)
        out = Dense(window_size // 2, activation='relu', name='encoder_dense3')(out)
        out = Dense(window_size // 4, activation='relu', name='encoder_dense4')(out)
        out = LSTM(96, activation='relu', kernel_regularizer=l2(), recurrent_regularizer=l2(), return_sequences=True, stateful=False)(out)
        out = Dense(window_size // 4, activation='relu', name='decoder_dense1')(out)
        out = Dense(window_size // 2, activation='relu', name='decoder_dense2')(out)
        out = Dense(window_size, activation='tanh', name='decoder_dense3')(out)
        out = Dense(window_size, activation='sigmoid', name='decoder_dense4')(out)
        output = multiply([out, main_input])

        model = Model(inputs=main_input, outputs=[output])
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=initial_lr), metrics=['mse'])

        model.summary()
        return model
