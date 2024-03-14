import keras.utils
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
if __name__ == '__main__':
    # Parametri
    num_rows = 1000
    num_cols = 5  # 5 colonne nella serie temporale di input
    seq_length = 20

    # Generazione della serie temporale
    data = np.random.rand(num_rows, num_cols)

    # Creazione del dataset X e Y
    X = []
    Y = []

    for i in range(num_rows - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length , 0])  # Selezioniamo solo la colonna 0 per il forecast

    X = np.array(X)
    Y = np.array(Y)

    # Creazione del modello LSTM
    model = Sequential([
        LSTM(64, input_shape=(seq_length, num_cols), return_sequences=True),
        LSTM(23, input_shape=(seq_length, num_cols), ),
        Dense(1)
    ])

    # Compilazione del modello
    model.compile(optimizer='adam', loss='mse')
    keras.utils.plot_model(model)
    model.summary()
    # Addestramento del modello
    model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)