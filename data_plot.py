"""
Dataset Plotting with scaling
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from data_generator import BaseDataset
from dataset import DatasetGenerator

if __name__ == '__main__':
    data_path = 'datasets/db18ecff/dataset'
    # dataset
    dataset = BaseDataset(data_path=data_path)
    # carico i dati, li divido e creo i generators
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    # li carico già divisi, non serve più splittarli
    train_generator, test_generator, _, __ = dataset.generate_data(train_filenames, test_filenames)
    # ottengo X_train,y_train ed X_test, y_test
    X_train, y_train = dataset.generator_to_Xy(train_generator)
    X_test, y_test = dataset.generator_to_Xy(test_generator)

    # unfold del dataset composto da (len(df['Rn_olb'])-30-1, 30, 12) elementi
    # in questo modo, rn conterrà len(df['Rn_olb']) valori, che corrispondono alle
    # misure di Radon da plottare
    # è un modo per passare da dataset alla colonna del dataframe
    # questo metodo è stato implementato in DatasetGenerator come
    # get_ts_from_ds()
    rn = DatasetGenerator.get_ts_from_ds(X_train, y_train, -2)
    df = pd.read_csv('data/train.csv')
    scaler = joblib.load('datasets/db18ecff/train-scalers/Rn_olb_scaler.save')
    scaled_rn = scaler.inverse_transform(rn.reshape(-1,1)).reshape(-2)

    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(scaled_rn, label = 'processed train dataset')
    plt.plot(df['Rn_olb'].values, label = 'train csv')
    plt.legend()
    plt.show()

    rn_test = DatasetGenerator.get_ts_from_ds(X_test, y_test, -2)
    df = pd.read_csv('data/test.csv')
    scaler = joblib.load('datasets/db18ecff/train-scalers/Rn_olb_scaler.save')
    scaled_rn_test = scaler.inverse_transform(rn_test.reshape(-1,1)).reshape(-2)
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(scaled_rn_test, label = 'processed test dataset')
    plt.plot(df['Rn_olb'].values,  label = 'test csv')
    plt.legend()
    plt.show()
