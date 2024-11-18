import pickle as pkl
from typing import List, Tuple

import keras
import numpy as np

from .libV2 import apply_filter


class CustomGenerator(keras.utils.Sequence):
    def __init__(self, filenames, batch_size, base_path='dataset/', shuffle=False, on_end_shuffle=True):
        super().__init__()
        self.X_filenames = filenames[:, 0]
        self.y_filenames = filenames[:, 1]
        self.batch_size = batch_size
        self.base_path = base_path
        self.indices = np.arange(filenames.shape[0])
        self.on_end_shuffle = on_end_shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return (np.ceil(len(self.X_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.X_filenames[inds]
        batch_y = self.y_filenames[inds]

        x_data = []
        y_data = []
        for x_filename in batch_x:
            with open(f'{self.base_path}/{x_filename}', 'rb') as data:
                x_data.append(pkl.load(data))
        for y_filename in batch_y:
            with open(f'{self.base_path}/{y_filename}', 'rb') as data:
                y_data.append(pkl.load(data))

        return np.array(x_data), np.array(y_data)

    def on_epoch_end(self):
        if self.on_end_shuffle:
            np.random.shuffle(self.indices)


class Operation:
    def __init__(self):
        pass

    def apply(self, x_data, y_data, old_y):
        return x_data, y_data, old_y


class FilterOperation(Operation):
    def __init__(self, a, target_col, forecast_col, filter):
        super().__init__()
        self.a = a
        self.target_col = target_col
        self.filter = filter
        self.forecast_col = forecast_col

    def apply(self, x_data, y_data, old_y):
        for s in range(x_data.shape[0]):
            for tg in self.target_col:
                xi = np.append(x_data[s, :, tg], y_data[s, :, :])
                yi = np.zeros(len(xi))
                if old_y is not None:
                    yi[0] = old_y[tg]
                filtered_x = apply_filter(xi, yi, self.a, self.filter)
                x_data[s, :, tg] = filtered_x[:-1]
                if tg in self.forecast_col:
                    y_data[s, :, :] = filtered_x[-1]
            old_y = x_data[s, 1]
        return x_data, y_data, old_y


class CustomOpsGenerator(CustomGenerator):
    def __init__(self, filenames, batch_size, base_path='dataset/', shuffle=False, on_end_shuffle=True,
                 operations=None):
        super().__init__(filenames, batch_size, base_path, shuffle, on_end_shuffle)
        self.operations: List[Operation] = operations if operations is not None else []
        self.old_y = None

    def __getitem__(self, idx):
        x_data, y_data = super().__getitem__(idx)
        x_data, y_data = self.apply_operations(x_data, y_data)
        return x_data, y_data

    def apply_operations(self, x_data, y_data):
        for operation in self.operations:
            x_data, y_data, old_y = operation.apply(x_data, y_data, self.old_y)
            self.old_y = old_y
        return x_data, y_data


class DataGenerator:
    def __init__(self, data_path=None, train_data_name='train_filenames.npy', test_data_name='test_filenames.npy'):
        self.data_path = data_path
        self.train_data_path = f'{self.data_path}/{train_data_name}'
        self.test_data_path = f'{self.data_path}/{test_data_name}'

    def __load_data(self, train_data, test_data, shuffle=False):
        # shuffle filenames
        if shuffle:
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            train_data = train_data[idx]
        return train_data, test_data

    def load_data(self, shuffle=False) -> Tuple[List[str], List[str]]:
        train_data, test_data = self.__load_data(np.load(self.train_data_path), np.load(self.test_data_path), shuffle)
        return train_data, test_data

    @staticmethod
    def split_train_valid(data, train_p=0.9, shuffle=False):
        # Implementazione della divisione dei dati
        # train_test split
        if type(data) is tuple:
            # print('tuple')
            X = data[0]
            y = data[1]
            if shuffle:
                idx = np.arange(len(X))
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]
            X_train, y_train = X[:int(len(X) * train_p)], y[:int(len(X) * train_p)]
            X_valid, y_valid = X[int(len(X) * train_p):], y[int(len(X) * train_p):]
            return (X_train, y_train), (X_valid, y_valid)
        else:
            print('generators')
            if shuffle:
                idx = np.arange(len(data))
                np.random.shuffle(idx)
                data = data[idx]
            train_filenames = data[:int(len(data) * train_p)]
            valid_filenames = data[int(len(data) * train_p):]
            return train_filenames, valid_filenames
        # Implementazione della divisione dei dati
        # train_test split

    def generate_data(self, train_filenames: List, test_filenames: List, batch_size=32, operations=None):
        # Implementazione della preparazione dei dati
        if operations is None:
            operations = []
        train_generator = CustomOpsGenerator(train_filenames, batch_size, base_path=self.data_path,
                                             operations=operations)
        test_generator = CustomOpsGenerator(test_filenames, batch_size, on_end_shuffle=False, base_path=self.data_path,
                                            operations=operations)
        example_generator = CustomOpsGenerator(train_filenames, batch_size, base_path=self.data_path,
                                               operations=operations)

        for idx, elem in enumerate(example_generator):
            if idx >= 1:
                break

        input_shape = (elem[0][0].shape[0], elem[0][0].shape[1])
        output_shape = (elem[1].shape[-1])
        return train_generator, test_generator, input_shape, output_shape

    @staticmethod
    def generator_to_Xy(generator):
        X, y = [], []
        for elem in generator:
            X.extend(elem[0])
            y.extend(elem[1])
        return np.array(X), np.array(y)
