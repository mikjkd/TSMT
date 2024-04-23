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


class ModelTrainer:
    def __init__(self, batch_size):
        self.batch_size: int = batch_size

    def train_model(self, config) -> History:
        i = config['input_shape']
        os = config['output_shape']
        model = config['model']
        model_name = config['model_name']
        len_train = config['len_train']
        len_test = config['len_test']
        train, test = config['train_generator'], config['test_generator']
        optimizer = config['optimizer']
        loss = config['loss']
        epochs = config['epochs']
        is_multiprocessing = config['multiprocessing']
        workers = 0 if not is_multiprocessing else config['workers']
        model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])

        plot_model(model)
        model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=60)
        mc = ModelCheckpoint(f'saved_model/{model_name}.keras', monitor='val_loss', mode='min',
                             verbose=1,
                             save_best_only=True)
        history = model.fit(x=train,
                            steps_per_epoch=int(len_train // self.batch_size),
                            validation_data=test,
                            validation_steps=int(len_test // self.batch_size),
                            epochs=epochs,
                            callbacks=[mc, es]
                            )

        return history

    def run(self, model, model_name, train, test, shapes):
        config = {
            'model': model,
            'model_name': model_name,
            'len_train': len(train['filenames']),
            'len_test': len(test['filenames']),
            'train_generator': train['generator'],
            'test_generator': test['generator'],
            'input_shape': shapes['input'],
            'output_shape': shapes['output'],
            'optimizer': Adam(),
            'loss': "mse",
            'epochs': 512,
            'multiprocessing': False
        }

        history = self.train_model(config)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend()
        plt.show()


class RegressorModel:
    def __init__(self, model_name):
        self.model: Optional[keras.Model] = None
        self.model_name: str = model_name
        self.history: Optional[History] = None

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        pass

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


# carico dati


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=64, return_sequences=False)(input1)
        out = Dense(output_shape)(l1)
        self.model = Model(inputs=input1, outputs=out)
        # return self.model


if __name__ == '__main__':  # model
    lstm_model_name = 'lstm_mae_model_with_valid'
    lstm_regressor = LSTMRegressor(model_name=lstm_model_name)
    # dataset
    dataset = BaseDataset(data_path='dataset')
    # trainer
    trainer = ModelTrainer(batch_size=64)

    # carico i dati, li divido e creo i generators
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    # li carico già divisi, non serve più splittarli
    train_filenames, valid_filenames = dataset.split_train_valid(train_filenames)
    train_generator, valid_generator, input_shape, output_shape = dataset.generate_data(train_filenames,
                                                                                        valid_filenames)

    # genero il modello a che prende in considerazione input ed output shape
    lstm_regressor.generate_model(input_shape, output_shape)

    # alleno il modello
    trainer.run(
        model=lstm_regressor.model,
        model_name=lstm_regressor.model_name,
        train={"filenames": train_filenames, "generator": train_generator},
        test={'filenames': valid_filenames, 'generator': valid_generator},
        shapes={'input': input_shape, 'output': output_shape}
    )

    _, test_generator, __, ___ = dataset.generate_data(train_filenames, valid_filenames)
    lstm_y_preds = lstm_regressor.model.predict(test_generator)
    lstm_regressor.model.evaluate(test_generator)
    # return lstm_regressor
