from typing import Optional

import joblib
import keras
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.layers import LSTM, Dense
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam
from keras.saving.save import load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt

from data_generator import CustomGenerator


class RegressorModel:
    def __init__(self, model_name, data_path):
        self.data_path: str = data_path
        self.model: Optional[keras.Model] = None
        self.model_name: str = model_name
        self.batch_size: int = 32
        self.history: Optional[History] = None

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        pass

    def load_data(self, shuffle=False):
        # Implementazione del caricamento dei dati
        data = np.load(self.data_path)
        # shuffle filenames
        if shuffle:
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            data = data[idx]
        return data

    def split_data(self, data, test_p=0.8, train_p=0.2):
        # Implementazione della divisione dei dati
        # train_test split
        train_filenames = data[:int(len(data) * test_p)]
        test_filenames = data[int(len(data) * test_p):-1]
        return train_filenames, test_filenames

    def generate_data(self, train_filenames, test_filenames, batch_size=32):
        # Implementazione della preparazione dei dati
        self.batch_size = batch_size
        train_generator = CustomGenerator(train_filenames, batch_size)
        test_generator = CustomGenerator(test_filenames, batch_size)
        return train_generator, test_generator

    def train_model(self, len_train, len_test, train_test_generator, config=None) -> History:
        if config is None:
            config = {
                'optimizer': Adam(),
                'loss': "mean_squared_error",
                'epochs': 64,
                'multiprocessing': False
            }

        train_generator, test_generator = train_test_generator[0], train_test_generator[1]
        for idx, elem in enumerate(train_generator):
            if idx >= 1:
                break

        i = (elem[0][0].shape[0], elem[0][0].shape[1])
        os = (elem[1].shape[-1])

        self.model = self.generate_model(input_shape=i, output_shape=os)

        plot_model(self.model)

        self.model.summary()

        es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1, patience=60)
        mc = ModelCheckpoint(f'saved_model/{self.model_name}.h5', monitor='val_mean_squared_error', mode='min',
                             verbose=1,
                             save_best_only=True)

        optimizer = config['optimizer']
        loss = config['loss']
        epochs = config['epochs']
        is_multiprocessing = config['multiprocessing']
        workers = 0 if not is_multiprocessing else config['workers']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['mean_squared_error'])

        history = self.model.fit(x=train_generator,
                                 steps_per_epoch=int(len_train // self.batch_size),
                                 validation_data=test_generator,
                                 validation_steps=int(len_test // self.batch_size),
                                 epochs=epochs,
                                 callbacks=[mc, es], use_multiprocessing=is_multiprocessing, workers=workers
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

    def run(self):
        pass


class LinearRegressor(RegressorModel):
    def generate_model(self, input_shape, output_shape):
        pass


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name, data_path):
        super().__init__(model_name, data_path)

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=8, return_sequences=False)(input1)
        out = Dense(output_shape)(l1)
        model = Model(inputs=input1, outputs=out)
        return model

    def run(self):
        # carico dati
        data = self.load_data(shuffle=True)
        # divido i dati e creo i generators
        train_filenames, test_filenames = self.split_data(data)
        train_generator, test_generator = self.generate_data(train_filenames, test_filenames)

        self.history = self.train_model(len(train_filenames), len(test_filenames), [train_generator, test_generator])

        plt.plot(self.history.history['loss'], label='last_time_step_mse')
        plt.plot(self.history.history['val_loss'], label='val_last_time_step_mse')
        plt.legend()
        plt.show()


"""
        input1 = keras.Input(shape=input_shape)
        lstm = LSTM(units=512, return_sequences=True)(input1)
        encoder_LSTM = LSTM(units=256, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM(lstm)
        decoder = RepeatVector(120)(encoder_outputs)
        concat = keras.layers.Concatenate(axis=2)([decoder, input2])
        decoder_outputs, _, _ = LSTM(256, return_state=True, return_sequences=True)(concat,
                                                                                    initial_state=[state_h, state_c])
        out = LSTM(512, return_sequences=True)(decoder_outputs)
        out = TimeDistributed(Dense(output_shape))(out)
        model = Model(inputs=input1, outputs=out)
        return model"""


# Esempio di utilizzo della classe ForecastModel

def scale_preds(preds, scaler_path):
    # Implementazione della predizione
    scaler = joblib.load(scaler_path)
    scaled_preds = []
    for p in preds:
        scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
    scaled_preds = [int(max(np.ceil(sp[0][0]), 0)) for sp in scaled_preds]
    return scaled_preds


if __name__ == '__main__':
    lstm_regressor = LSTMRegressor(model_name='lstm_model', data_path='dataset/filenames.npy')

    # lstm_regressor.run()   ALREADY TRAINED

    lstm_regressor.load_model('saved_model/lstm_model.h5')
    data = lstm_regressor.load_data(shuffle=False)
    # divido i dati e creo i generators
    train_filenames, test_filenames = lstm_regressor.split_data(data)
    _, test_generator = lstm_regressor.generate_data(train_filenames, test_filenames)

    y_preds = lstm_regressor.model.predict(test_generator)

    # devo estrarre le y dai generators
    y_true = []
    for y in test_generator:
        y_true.extend(y[1][:, 0, 0])
    y_true = np.array(y_true)

    lstm_regressor.evaluate_model(y_preds.reshape(y_preds.shape[0], ), y_true)

    scaled_y = scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')
    scaled_true = scale_preds(y_true, scaler_path='scalers/Rn_olb_scaler.save')

    for z in zip(scaled_y, scaled_true):
        print(z)
