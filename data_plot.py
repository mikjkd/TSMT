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
    X, y = dataset_generator.generate_XY(base_path, columns_to_scale=[],
                                         columns_to_drop=[],
                                         columns_to_forecast=['Rn_olb'],
                                         save=False, cast_values=False, remove_not_known=True)

    # prendo i valori del radon per Olibano
    df['date'] = pd.to_datetime(df['date'])
    x_vals = df['date'].dt.strftime("%d-%m-%y").values
    y_vals = df['Rn_olb'].values

    thr = 80000

    t_points = np.where(y_vals < thr)
    y_t = y_vals.copy()
    y_t[t_points] = np.nan

    for z in zip(y_vals, y_t):
        print(z)

    thr_vals = np.ones(len(y_vals))
    thr_vals = thr * thr_vals

    x_thr_val = len(y_vals) * 0.8

    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(x_vals, y_vals)
    plt.axvline(x=x_thr_val, color='y')
    plt.scatter(x_vals, y_t)
    plt.plot(x_vals, thr_vals, '-b')
    plt.show()
