import hashlib
import string
from random import random
from typing import Optional

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.losses import mean_squared_error, mean_absolute_error
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.saving.saving_api import load_model


class ModelTrainer:
    def __init__(self, batch_size):
        self.batch_size: int = batch_size

    def train_model(self, config) -> History:
        model = config['model']
        model_name = config['model_name']
        len_train = config['len_train']
        len_test = config['len_test']
        train_gen, valid_gen = config['train_generator'], config['valid_generator']
        train, valid = config['train'], config['valid']
        optimizer = config['optimizer']
        loss = config['loss']
        epochs = config['epochs']
        is_multiprocessing = config['multiprocessing']
        workers = 0 if not is_multiprocessing else config['workers']
        model.compile(loss=loss, optimizer=optimizer, metrics=loss)

        # plot_model(model)
        model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        mc = ModelCheckpoint(f'saved_model/{model_name}.x', monitor='val_loss', mode='min',
                             verbose=1,
                             save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=0.00001)
        # if generator
        if train_gen is not None and valid_gen is not None:
            history = model.fit(x=train_gen,
                                steps_per_epoch=int(len_train // self.batch_size),
                                validation_data=valid_gen,
                                validation_steps=int(len_test // self.batch_size),
                                epochs=epochs,
                                callbacks=[mc, es, reduce_lr]
                                )
        elif train is not None and valid is not None:
            X_train, y_train = train['X'], train['y']
            X_valid, y_valid = valid['X'], valid['y']
            history = model.fit(x=X_train, y=y_train,
                                validation_data=(X_valid, y_valid),
                                batch_size=self.batch_size,
                                epochs=epochs,
                                callbacks=[mc, es, reduce_lr]
                                )
        else:
            raise Exception('Wrong configuration on training')
        return history

    def run(self, model, model_name, train, valid, optimizer=Adam(learning_rate=0.001), loss='mse', epochs=512):
        config = {
            'model': model,
            'model_name': model_name,
            'len_train': len(train['filenames']),
            'len_test': len(valid['filenames']),
            'train_generator': train['generator'] if 'generator' in train else None,
            'valid_generator': valid['generator'] if 'generator' in valid else None,
            'train': train['data'] if 'data' in train else None,
            'valid': valid['data'] if 'data' in valid else None,
            'optimizer': optimizer,
            'loss': loss,
            'epochs': epochs,
            'multiprocessing': False
        }

        history = self.train_model(config)

        plt.plot(history.history['loss'], label=f"{config['loss']}")
        plt.plot(history.history['val_loss'], label=f"val_{config['loss']}")
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
        y_pred = y_pred.reshape(-1)
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


def generate_model_name():
    # Convert hyperparameters to a string
    letters = string.ascii_lowercase  # Use lowercase letters
    hyperparameters = ''.join(random.choice(letters) for i in range(20))

    hyperparameters_str = str(hyperparameters) + str(random.randint(1, 1000))
    # Generate SHA-256 hash
    hash_object = hashlib.sha256(hyperparameters_str.encode())
    model_name = hash_object.hexdigest()[:8]  # Take first 8 characters for readability

    return model_name
