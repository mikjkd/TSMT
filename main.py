import numpy as np

from model import LSTMRegressor

if __name__ == '__main__':
    # Parametri
    num_rows = 1000
    num_cols = 5  # 5 colonne nella serie temporale di input
    seq_length = 20

    # Generazione della serie temporale
    data = np.random.rand(num_rows, num_cols)

    ### DA IMPLEMENTARE BY CLASS
    # Creazione del dataset X e Y
    X = []
    Y = []

    for i in range(num_rows - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length, 0])  # Selezioniamo solo la colonna 0 per il forecast

    X = np.array(X)
    Y = np.array(Y)

    # Forecast Model
    lstm_regressor = LSTMRegressor(model_name='lstm_model', data_path='', seq_len=seq_length, num_cols=num_cols)
    lstm_regressor.run()