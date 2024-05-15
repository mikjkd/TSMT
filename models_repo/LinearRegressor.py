import keras
from model import RegressorModel


class LinearRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        model = keras.Sequential()
        # Aggiungi un layer Flatten per linearizzare il tensore
        model.add(keras.layers.Flatten(input_shape=input_shape))
        # Aggiungi il layer Dense successivo
        model.add(keras.layers.Dense(units=output_shape))  # specifica il numero di unità del layer Dense
        self.model = model
