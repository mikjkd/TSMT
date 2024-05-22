import keras
from keras import Model
from keras.src.layers import LSTM, Dense

from model import RegressorModel


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=64, return_sequences=False)(input1)
        out = Dense(output_shape, activation='relu')(l1)
        self.model = Model(inputs=input1, outputs=out)
        # return self.model


class LSTMRegressor2L(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=50, return_sequences=True)(input1)
        l2 = LSTM(units=50, return_sequences=False)(l1)
        out = Dense(output_shape)(l2)
        self.model = Model(inputs=input1, outputs=out)
        # return self.model
