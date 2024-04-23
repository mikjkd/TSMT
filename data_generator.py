import pickle as pkl

import keras
import numpy as np


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
            with open(self.base_path + x_filename, 'rb') as data:
                x_data.append(pkl.load(data))
        for y_filename in batch_y:
            with open(self.base_path + y_filename, 'rb') as data:
                y_data.append(pkl.load(data))

        return np.array(x_data), np.array(y_data)

    def on_epoch_end(self):
        if self.on_end_shuffle:
            np.random.shuffle(self.indices)


class BaseDataset:
    def __init__(self, data_path, train_data_name='train_filenames.npy', test_data_name='test_filenames.npy'):
        self.data_path = data_path
        self.train_data_path = f'{self.data_path}/{train_data_name}'
        self.test_data_path = f'{self.data_path}/{test_data_name}'

    def load_data(self, shuffle=False):
        # Implementazione del caricamento dei dati
        train_data = np.load(self.train_data_path)
        test_data = np.load(self.test_data_path)
        # shuffle filenames
        if shuffle:
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            train_data = train_data[idx]
        return train_data, test_data

    """
        deprecato: il dataset prima era unico, attualmente è già diviso in train e test
    """

    def split_train_valid(self, data, train_p=0.9):
        # Implementazione della divisione dei dati
        # train_test split
        train_filenames = data[:int(len(data) * train_p)]
        valid_filenames = data[int(len(data) * train_p):-1]
        return train_filenames, valid_filenames

    def generate_data(self, train_filenames, test_filenames, batch_size=32):
        # Implementazione della preparazione dei dati
        train_generator = CustomGenerator(train_filenames, batch_size)
        test_generator = CustomGenerator(test_filenames, batch_size, on_end_shuffle=False)
        example_generator = CustomGenerator(train_filenames, batch_size)

        for idx, elem in enumerate(example_generator):
            if idx >= 1:
                break

        input_shape = (elem[0][0].shape[0], elem[0][0].shape[1])
        output_shape = (elem[1].shape[-1])
        return train_generator, test_generator, input_shape, output_shape


class KFoldDataset(BaseDataset):
    def __init__(self, data_path):
        super(KFoldDataset, self).__init__(data_path)

    def generate_data(self, train_filenames, test_filenames, batch_size=32, k=5):
        pass
