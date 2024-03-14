from typing import Optional

import keras
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt

from data_generator import CustomGenerator
from libV2 import last_time_step_mse


class RegressorModel:
    def __init__(self, model_name, data_path):
        self.data_path: str = data_path
        self.model: Optional[keras.Model] = None
        self.model_name: str = model_name
        self.batch_size: int = 32
        self.history: Optional[History] = None

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        pass

    def load_data(self):
        # Implementazione del caricamento dei dati
        data = np.load(self.data_path)
        # shuffle filenames
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
                'loss': "mse",
                'epochs': 64,
                'multiprocessing': False
            }

        train_generator, test_generator = train_test_generator[0], train_test_generator[1]
        for idx, elem in enumerate(train_generator):
            if idx >= 1:
                break

        i = (elem[0][0].shape[1], elem[0][0].shape[2])
        os = (elem[1].shape[-1])

        self.model = self.generate_model(input_shape=i, output_shape=os)

        plot_model(self.model)

        self.model.summary()

        es = EarlyStopping(monitor='val_last_time_step_mse', mode='min', verbose=1, patience=60)
        mc = ModelCheckpoint(self.model_name, monitor='val_last_time_step_mse', mode='min', verbose=1,
                             save_best_only=True)

        optimizer = config['optimizer']
        loss = config['loss']
        epochs = config['epochs']
        is_multiprocessing = config['multiprocessing']
        workers = 0 if not is_multiprocessing else config['workers']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[last_time_step_mse])

        history = self.model.fit(x=train_generator,
                                 steps_per_epoch=int(len_train // self.batch_size),
                                 validation_data=test_generator,
                                 validation_steps=int(len_test // self.batch_size),
                                 epochs=epochs,
                                 callbacks=[mc, es], use_multiprocessing=is_multiprocessing, workers=workers
                                 )

        return history

    def evaluate_model(self):
        # Implementazione della valutazione del modello
        pass

    def make_predictions(self, new_data):
        # Implementazione della predizione
        pass

    def visualize_results(self, actual, predicted):
        # Implementazione della visualizzazione dei risultati
        pass

    def run(self):
        pass


class LinearRegressor(RegressorModel):
    def generate_model(self, input_shape, output_shape):
        pass


class LSTMRegressor(RegressorModel):
    def __init__(self, model_name, data_path, seq_len, num_cols):
        super().__init__(model_name, data_path)
        self.seq_len = seq_len
        self.num_cols = num_cols

    def generate_model(self, input_shape, output_shape):
        input1 = keras.Input(shape=input_shape)
        l1 = LSTM(units=128, return_sequences=True)(input1)
        l2 = LSTM(64, input_shape=(self.seq_len, self.num_cols), return_sequences=True)(l1)
        l3 = LSTM(23, input_shape=(self.seq_len, self.num_cols), )(l2)
        out = Dense(1)(l3)
        model = Model(inputs=input1, outputs=out)
        return model

    def run(self):
        # carico dati
        data = self.load_data()
        # divido i dati e creo i generators
        train_filenames, test_filenames = self.split_data(data)
        train_generator, test_generator = self.generate_data(train_filenames, test_filenames)

        self.history = self.train_model(len(train_filenames), len(test_filenames), [train_generator, test_generator])

        plt.plot(self.history.history['loss'], label='last_time_step_mse')
        plt.plot(self.history.history['val_loss'], label='val_last_time_step_mse')
        plt.legend()


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
