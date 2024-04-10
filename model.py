from typing import Optional

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.layers import LSTM, Dense
from keras.losses import mean_squared_error, mean_absolute_error
from keras.src.optimizers import Adam
from keras.src.saving.saving_api import load_model
from keras.utils import plot_model

from data_generator import BaseDataset


class RegressorModel:
    def __init__(self, model_name, batch_size=32):
        self.model: Optional[keras.Model] = None
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.history: Optional[History] = None

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        pass

    def train_model(self, len_train, len_test, train_test_generator, data_shape, config=None) -> History:
        if config is None:
            config = {
                'optimizer': Adam(learning_rate=1e-5),
                'loss': "mse",
                'epochs': 512,
                'multiprocessing': False
            }

        i = data_shape[0]
        os = data_shape[1]

        train, test = train_test_generator[0], train_test_generator[1]

        self.model = self.generate_model(input_shape=i, output_shape=os)

        plot_model(self.model)

        self.model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=60)
        mc = ModelCheckpoint(f'saved_model/{self.model_name}.keras', monitor='val_loss', mode='min',
                             verbose=1,
                             save_best_only=True)

        optimizer = config['optimizer']
        loss = config['loss']
        epochs = config['epochs']
        is_multiprocessing = config['multiprocessing']
        workers = 0 if not is_multiprocessing else config['workers']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])

        history = self.model.fit(x=train,
                                 steps_per_epoch=int(len_train // self.batch_size),
                                 validation_data=test,
                                 validation_steps=int(len_test // self.batch_size),
                                 epochs=epochs,
                                 callbacks=[mc, es]
                                 )

        return history

    def load_model(self, model_path):
        self.model = load_model(model_path)
        return self.model

    def evaluate_model(self, y_pred, y_true):
        # Implementazione della valutazione del modello
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("Root Mean Squared Error (RMSE):", rmse)

    def make_predictions(self, X, scaler_path):
        # Implementazione della predizione
        scaler = joblib.load(scaler_path)
        preds = self.model.predict(X)
        scaled_preds = []
        for p in preds[:, 0]:
            scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
        scaled_preds = [int(max(np.ceil(sp[0][0]), 0)) for sp in scaled_preds]
        return scaled_preds

    def visualize_results(self, actual, predicted):
        # Implementazione della visualizzazione dei risultati
        pass

    def run(self, train, test, shapes):
        self.history = self.train_model(len(train['filenames']), len(test['filenames']),
                                        [train['generator'], test['generator']],
                                        data_shape=[shapes['input'], shapes['output']])

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend()
        plt.show()


class LinearRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        model = keras.Sequential()
        # Aggiungi un layer Flatten per linearizzare il tensore
        model.add(keras.layers.Flatten(input_shape=input_shape))
        # Aggiungi il layer Dense successivo
        model.add(keras.layers.Dense(units=output_shape))  # specifica il numero di unità del layer Dense
        return model


# carico dati


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=64, return_sequences=False)(input1)
        out = Dense(output_shape)(l1)
        model = Model(inputs=input1, outputs=out)
        return model


def model(model_name):
    lstm_model_name = model_name
    lstm_regressor = LSTMRegressor(model_name=lstm_model_name, )
    dataset = BaseDataset(data_path='dataset/filenames.npy')
    data = dataset.load_data(shuffle=False)
    # divido i dati e creo i generators
    train_filenames, test_filenames = dataset.split_data(data)
    train_generator, test_generator, input_shape, output_shape = dataset.generate_data(train_filenames, test_filenames)

    lstm_regressor.run(
        {"filenames": train_filenames, "generator": train_generator},
        {'filenames': test_filenames, 'generator': test_generator},
        {'inout': input_shape, 'output': output_shape}
    )

    lstm_y_preds = lstm_regressor.model.predict(test_generator)
    lstm_regressor.model.evaluate(test_generator)
    return lstm_regressor


if __name__ == '__main__':
    name = 'lstm_model_no_zeros'
    model(name)
