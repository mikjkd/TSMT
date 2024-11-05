import keras
from keras import Model
from keras.src.layers import LSTM, Dense, TimeDistributed

from model import RegressorModel


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def description(self):
        return 'LSTM Model with one layer (64 units) a Dense (1 linear output)'

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=64, return_sequences=False)(input1)
        out = Dense(output_shape, activation='linear')(l1)
        self.model = Model(inputs=input1, outputs=out)
        return self.model


class TDLSTMRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def description(self):
        return 'LSTM Model with one layer (64 units) a Distributed Dense'

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        # First LSTM layer with more units and dropout for regularization
        l1 = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input1)
        # Second LSTM layer for further processing
        l2 = LSTM(units=64, return_sequences=True, dropout=0.2)(l1)
        # Third LSTM layer to capture more sequence memory
        l3 = LSTM(units=32, return_sequences=True)(l2)
        # TimeDistributed Dense layer for per-timestep output
        out = TimeDistributed(Dense(output_shape, activation='linear'))(l3)
        self.model = Model(inputs=input1, outputs=out)
        return self.model


class LSTMRegressor2L(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        # First LSTM layer with more units and dropout for regularization
        l1 = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input1)
        # Second LSTM layer for further processing
        l2 = LSTM(units=64, return_sequences=True, dropout=0.2)(l1)
        # Third LSTM layer to capture more sequence memory
        l3 = LSTM(units=32, return_sequences=False)(l2)
        # TimeDistributed Dense layer for per-timestep output
        out = Dense(output_shape, activation='linear')(l3)
        self.model = Model(inputs=input1, outputs=out)
        return self.model
