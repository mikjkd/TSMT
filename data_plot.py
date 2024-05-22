"""
Dataset Plotting with scaling
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from math import floor

from data_generator import BaseDataset
from dataset import DatasetGenerator
from libV2 import fill_na_mean

if __name__ == '__main__':
    data_path = 'data'
    dataset_path = 'dataset'
    train_scalers_path = 'train-scalers'
    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']
    """
        In questo punto carico i dati con i generator e li trasformo in X,y
    """
    # dataset
    dataset = BaseDataset(data_path=dataset_path)
    # carico i dati, li divido e creo i generators
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    # li carico già divisi, non serve più splittarli
    train_generator, test_generator, _, __ = dataset.generate_data(train_filenames, test_filenames)
    # ottengo X_train,y_train ed X_test, y_test
    X_train, y_train = dataset.generator_to_Xy(train_generator)
    X_test, y_test = dataset.generator_to_Xy(test_generator)
    """
        qui carico i dati dal csv direttamente
        riempio i buchi con fill_na_mean 
        faccio questa operazione per confrontare poi i dati X,y inversamente scalati, con i dati reali
        
    
    full_df = pd.read_csv(f'{data_path}/olb_msa_full.csv')
    full_df.columns = columns
    filled_df = fill_na_mean(full_df,columns)

    # stampo i valori di interesse prima normali, poi riempiti
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(filled_df['Rn_olb'], label = 'filled')
    plt.plot(full_df['Rn_olb'], label='normal')
    plt.legend()
    plt.show()

    # unfold del dataset composto da (len(df['Rn_olb'])-30-1, 30, 12) elementi
    # in questo modo, rn conterrà len(df['Rn_olb']) valori, che corrispondono alle
    # misure di Radon da plottare
    # è un modo per passare da dataset alla colonna del dataframe
    # questo metodo è stato implementato in DatasetGenerator come
    # get_ts_from_ds()
    rn = DatasetGenerator.get_ts_from_ds(X_train, y_train, -2)
    #df = pd.read_csv(f'{data_path}/train.csv')
    scaler = joblib.load(f'{train_scalers_path}/Rn_olb_scaler.save')
    scaled_rn = scaler.inverse_transform(rn.reshape(-1,1)).reshape(-2)

    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(scaled_rn, label = 'processed train dataset')
    # plotto l'80% del dataframe
    plt.plot(full_df['Rn_olb'].values[:floor(len(full_df)*0.8)], label = 'train csv')
    plt.legend()
    plt.show()

    rn_test = DatasetGenerator.get_ts_from_ds(X_test, y_test, -2)
    #df = pd.read_csv(f'{data_path}/test.csv')
    scaler = joblib.load(f'{train_scalers_path}/Rn_olb_scaler.save')
    scaled_rn_test = scaler.inverse_transform(rn_test.reshape(-1,1)).reshape(-2)
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(scaled_rn_test, label = 'processed test dataset')
    plt.plot(full_df['Rn_olb'].values[floor(len(full_df)*0.8):], label = 'test csv')
    plt.legend()
    plt.show()
    """
    scaler_names = ['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb']

    scalers= [joblib.load(f'scalers/{s}_scaler.save') for s in scaler_names]
    for i in range(X_train.shape[2]):
        rn = DatasetGenerator.get_ts_from_ds(X_train, i)
        rn_scaled = scalers[i].inverse_transform(rn.reshape(-1, 1)).reshape(len(rn))
        fig, axs = plt.subplots(2,figsize=(20, 6))
        axs[0].plot(rn)
        axs[1].plot(rn_scaled)
        plt.show()

