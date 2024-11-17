import keras
from keras import Model
from keras.src.layers import LSTM, Dense, TimeDistributed, MultiHeadAttention, Flatten, Concatenate, Conv1D, \
    LayerNormalization

from .model import RegressorModel


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


class TFTModel(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)

    def description(self):
        return "TFT model implementation"

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        inputs = keras.Input(shape=input_shape)

        # LSTM Encoder
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = LSTM(32)(lstm_out)

        # Multi-Head Attention for feature interaction
        attention_out = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        attention_out = Flatten()(attention_out)

        # Concatenate LSTM and Attention outputs
        combined = Concatenate()([lstm_out, attention_out])

        # Fully Connected Layers
        dense1 = Dense(64, activation="relu")(combined)
        dense2 = Dense(32, activation="relu")(dense1)

        # Final output layer
        outputs = Dense(output_shape)(dense2)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model


class Autoformer(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)

    def description(self):
        return "Autoformer implementation"

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        def autoformer(input_shape, output_shape, num_heads=4, num_layers=2, d_model=64):
            # Define input layer
            inputs = keras.Input(shape=input_shape)

            # Trend Decomposition
            trend = Conv1D(filters=d_model, kernel_size=3, padding="same")(inputs)
            trend = Dense(d_model, activation="relu")(trend)

            # Seasonal Decomposition
            seasonal = Conv1D(filters=d_model, kernel_size=3, padding="same")(inputs)
            seasonal = Dense(d_model, activation="relu")(seasonal)

            # Transformer Encoder-Decoder Layers
            for _ in range(num_layers):
                trend = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(trend, trend)
                trend = LayerNormalization(epsilon=1e-6)(trend)

                seasonal = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(seasonal, seasonal)
                seasonal = LayerNormalization(epsilon=1e-6)(seasonal)

            # Recompose trend and seasonal components
            combined = Concatenate()([trend, seasonal])
            combined = Dense(output_shape)(combined)

            # Final model output
            return keras.Model(inputs=inputs, outputs=combined)

        self.model = autoformer(input_shape=input_shape, output_shape=output_shape)
        return self.model
