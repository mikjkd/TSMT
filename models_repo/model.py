import hashlib
import random
import string
from enum import Enum
from typing import Optional

import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.saving.saving_api import load_model


class ModelTrainer:
    def __init__(self, batch_size):
        self.batch_size: int = batch_size

    # @TODO: Train model with k-fold
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
            'len_train': len(train['filenames']) if 'filenames' in train else None,
            'len_test': len(valid['filenames']) if 'filenames' in valid else None,
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


class PredMode(Enum):
    STD = 'Standard'
    FR = 'FreeRun'


class PredictConfig:
    def __init__(self, pred_mode: PredMode, options=None):
        self.mode = pred_mode
        self.options = options


class RegressorModel:
    def __init__(self, model_name, model=None, history=None):
        self.model: Optional[keras.Model] = model
        self.model_name: str = model_name
        self.history: Optional[History] = history
        self.pred_config: PredictConfig = PredictConfig(PredMode.STD)

    def generate_model(self, input_shape, output_shape) -> keras.Model:
        pass

    def set_pred_mode(self, pred_mode: PredMode, options=None):
        self.pred_config.mode = pred_mode
        if pred_mode is PredMode.FR:
            self.pred_config.options = options

    def load_model(self, model_path):
        self.model = load_model(model_path)
        return self.model

    def predict(self, X):
        # Implementazione della predizione
        if self.pred_config.mode is PredMode.STD:
            preds = self.model.predict(X)
        elif self.pred_config.mode is PredMode.FR:
            raise Exception('Free-Run not already implemented')
            # steps = self.pred_config.options['steps']
            # preds = []
            # v = None
            # for i in range(0, steps):
            #    X_input = X[i].reshape(1, X.shape[1], X.shape[2])
            #    if i > 0:
            #        idx = min(i, 30)
            #        X_input[0, -idx:, 4] = preds[-idx:]
            #    v = self.model.predict(X_input)
            #    preds.append(v.item())
            # preds = np.array(preds)
            # preds = preds.reshape(preds.shape[0], 1)
        else:
            raise Exception('Wrong prediction execution')
        return preds

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
