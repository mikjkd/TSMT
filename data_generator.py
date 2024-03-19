import pickle as pkl
import numpy as np
import keras


class CustomGenerator(keras.utils.Sequence):
    def __init__(self, filenames, batch_size, base_path='dataset/'):
        self.X_filenames = filenames[:, 0]
        self.y_filenames = filenames[:, 1]
        self.batch_size = batch_size
        self.base_path = base_path
        self.indices = np.arange(filenames.shape[0])
        np.random.shuffle(self.indices)

    def __len__(self):
        return (np.ceil(len(self.X_filenames)) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        # idx*self.batch_size: (idx+1)*self.batch_size
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
        return [np.array(x_data), np.array(y_data)]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
