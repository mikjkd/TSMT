import numpy as np

from dataset import DatasetGenerator
from model import LSTMRegressor

if __name__ == '__main__':
    # Parametri
    num_rows = 1000
    num_cols = 5  # 5 colonne nella serie temporale di input
    seq_length = 20

    # Generazione della serie temporale
    data = np.random.rand(num_rows, num_cols)

    data_path = 'data/olb_msa_full.csv'
    base_path = 'dataset/'
    encoders = 'encoders/'
    scalers = 'scalers/'
    seq_len_x = 7
    seq_len_y = 1

    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']

    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)

    dataset_generator.generate_XY(columns_to_scale=['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
                                                    'Ru_msa', 'P_msa', 'Rn_msa'],
                                  columns_to_drop=['date', 'displacement (cm)',
                                                   'background seismicity'],
                                  columns_to_forecast=['Rn_olb'])

    # Forecast Model
    lstm_regressor = LSTMRegressor(model_name='lstm_model', data_path='dataset/', seq_len=seq_length, num_cols=num_cols)
    lstm_regressor.run()