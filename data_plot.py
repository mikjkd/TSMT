"""
Dataset Plotting with scaling
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import DatasetGenerator

if __name__ == '__main__':
    data_path = 'data/olb_msa_full.csv'
    base_path = 'dataset/'
    encoders = 'encoders/'
    scalers = 'scalers/'

    if not os.path.exists(base_path):
        os.mkdir(base_path)
        print(f'{base_path} creata')

    if not os.path.exists(encoders):
        os.mkdir(encoders)
        print(f'{encoders} creata')

    if not os.path.exists(scalers):
        os.mkdir(scalers)
        print(f'{scalers} creata')

    seq_len_x = 30
    seq_len_y = 1

    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']

    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    df = dataset_generator.generate_frame(fill_na=False)
    X, y = dataset_generator.generate_XY(columns_to_scale=[],
                                         columns_to_drop=[],
                                         columns_to_forecast=['Rn_olb'],
                                         cast_values=False, remove_not_known=False)

    # unfold del dataset composto da (len(df['Rn_olb'])-30-1, 30, 12) elementi
    # in questo modo, rn conterrà len(df['Rn_olb']) valori, che corrispondono alle
    # misure di Radon da plottare
    # è un modo per passare da dataset alla colonna del dataframe
    # questo metodo è stato implementato in DatasetGenerator come
    # get_ts_from_ds()
    rn = X[:, 0, 5]
    rn = np.append(rn[:-1], X[-1, :, 5])
    rn = np.append(rn, y[-1])

    # prendo i valori del radon per Olibano
    df['date'] = pd.to_datetime(df['date'])
    x_vals = df['date'].dt.strftime("%d-%m-%y").values
    y_vals = df['Rn_olb'].values
    fig, axis = plt.subplots(2, 1, figsize=(20, 6), dpi=80)
    axis[0].plot(x_vals, y_vals)
    axis[1].plot(x_vals, rn)
    plt.show()

    # definisco la threshold sopra la quale plottare i puntini
    thr = 80000

    t_points = np.where(y_vals < thr)
    y_t = y_vals.copy()
    # creo un vettore di lunghezza y_vals, che ha nan sui valori
    # che sono sotto thr e hanno il valore reale per gli altri
    y_t[t_points] = np.nan

    # creo la retta y = thr
    thr_vals = np.ones(len(y_vals))
    thr_vals = thr * thr_vals

    # trovo il punto x che indica l'80% del dataset
    x_thr_val = len(y_vals) * 0.8

    plt.figure(figsize=(20, 6), dpi=80)
    # stampo i valori
    plt.plot(x_vals, y_vals)
    # stampo la retta verticale che divide il dataset all'80%
    plt.axvline(x=x_thr_val, color='y')
    # stampo i puntini sui valori che superano la soglia
    plt.scatter(x_vals, y_t)
    # stampo la retta orizzontale y = thr
    plt.plot(x_vals, thr_vals, '-b')
    plt.show()
