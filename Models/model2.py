from DroneDenoise.Models.Configuration import *
from keras.layers import Dense, Input, multiply, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers


class Model2(object):
    def build_model(self):
        print('Build model...')
        main_input = Input(shape=(None, window_size), name='main_input')
        out = Dense(window_size // 2, activation='relu', name='encoder_dense1')(main_input)  # output shape: 480
        out = Dense(window_size // 4, activation='relu', name='encoder_dense2')(out)         # output shape: 240
        out = Dense(window_size // 8, activation='relu', name='encoder_dense3')(out)         # output shape: 120
        out = Dense(window_size // 16, activation='relu', name='encoder_dense4')(out)        # output shape: 60
        out = LSTM(60, activation='relu', kernel_regularizer=l2(), recurrent_regularizer=l2(), return_sequences=True, stateful=False)(out)
        out = Dense(window_size // 8, activation='relu', name='decoder_dense1')(out)         # output shape: 120
        out = Dense(window_size // 4, activation='relu', name='decoder_dense2')(out)         # output shape: 240
        out = Dense(window_size // 2, activation='relu', name='decoder_dense3')(out)         # output shape: 480
        out = Dense(window_size     , activation='sigmoid', name='decoder_dense4')(out)      # output shape: 960
        output = multiply([out, main_input])

        model = Model(inputs=main_input, outputs=[output])
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=initial_lr), metrics=['mse'])

        model.summary()
        return model
