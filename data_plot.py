"""
Dataset Plotting with scaling
"""
import os

import matplotlib.pyplot as plt

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
                                         save=False, cast_values=False)

    # prendo i valori del radon per Olibano
    y_vals = df['Rn_olb'].values
    plt.figure(figsize=(16, 8), dpi=80)
    plt.plot(y_vals)
    plt.show()
