import csv
import os
import pickle as pkl
from datetime import datetime, timedelta
from enum import Enum
from math import floor, ceil

import joblib
import numpy as np
import pandas as pd

from libV2 import minMaxScale, standardScale, split_sequence, fill_na_mean
from sklearn.model_selection import train_test_split


class ScalerTypes(Enum):
    MINMAX = 'minmax'
    STD = 'standard'


class FillnaTypes(Enum):
    SIMPLE = 'simple'
    MEAN = 'mean'

    @staticmethod
    def from_string(s):
        if s == 'SIMPLE':
            return FillnaTypes.SIMPLE
        elif s == 'MEAN':
            return FillnaTypes.MEAN
        else:
            return FillnaTypes.SIMPLE


class DatasetGenerator:
    def __init__(self, columns, seq_len_x, seq_len_y, data_path, encoders, scaler_path):
        self.columns = columns
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.data_path = data_path
        self.encoders = encoders
        self.scaler_path = scaler_path

    def scale_df(self, frame, columns_to_scale=None, scaler_names=None,
                 scalerType: ScalerTypes = ScalerTypes.MINMAX):
        if columns_to_scale is None:
            print('No columns to scale')
            return
        frame = frame.copy()
        if scaler_names:
            # ho già gli scalers, li carico e li utilizzo
            scalers = []
            for scaler_name in scaler_names:
                scalers.append(joblib.load(scaler_name))
            for idx, cts in enumerate(columns_to_scale):
                frame[cts] = scalers[idx].transform(
                    frame[cts].values.reshape(-1, 1).astype('float64'))
        else:
            # creo gli scalers in base al tipo e li salvo
            for cts in columns_to_scale:
                if scalerType == ScalerTypes.MINMAX:
                    tmp_scaler = minMaxScale(frame, cts)
                else:
                    tmp_scaler = standardScale(frame, cts)
                # salvo lo scaler
                joblib.dump(tmp_scaler, self.scaler_path + cts + '_scaler.save')
        return frame

    def load_XY(self):
        pass

    def __process_ds(self, frame, date_prediction_start=None, days_delta=7):
        frame = frame.copy()

        if date_prediction_start is not None:
            date_prediction_start = datetime.strptime(
                date_prediction_start, "%Y-%m-%d")
            date_ts_start = date_prediction_start - timedelta(days=days_delta)
            frame = frame[frame['date'] >= date_ts_start.strftime("%Y-%m-%d")]
            frame.reset_index(inplace=True, drop=True)
        return frame

    def generate_frame(self, start_date=None, end_date=None):
        df = pd.read_csv(self.data_path)
        df.columns = self.columns

        if start_date:
            df = df[df['date'] >= start_date]

        if end_date:
            df = df[df['date'] <= end_date]

        return df

    def generate_XY(self, columns_to_scale, columns_to_drop, columns_to_forecast, start_date=None,
                    end_date=None, cast_values=True, remove_not_known=False,
                    fill_na_type: FillnaTypes = FillnaTypes.SIMPLE):
        # frame contiene le informazioni passate e viene processato dalla rete per creare delle predizioni
        # info frame contiene le informazioni che la rete sfrutta per migliorare le predizioni

        # merge dataset
        df = self.generate_frame(start_date=start_date, end_date=end_date)

        ## filling na part -> A differenza del dataframe che può contenere valori vuoti (NaN), la sequenza (X,y) non può avere
        # valori NA
        # it is mandatory to fill na values in a (X,y) generation, it oly depends on the way
        # knowing the fact that we need to fill, a good way is to add a column to the df, saying that a specific value
        # has been filled with another value. That helps in post processing part

        # Adding the flag to the NAs columns
        na_cols = np.isnan(df[columns_to_forecast[0]].values)
        df = df.assign(na_cols=na_cols)

        if fill_na_type == FillnaTypes.SIMPLE:
            df = df.fillna(0)
        elif fill_na_type == FillnaTypes.MEAN:
            df = fill_na_mean(df, self.columns)

        # scalo le features e rimuovo quelle inutili
        frame = self.scale_df(df, columns_to_scale=columns_to_scale, scalerType=ScalerTypes.MINMAX)
        frame_drop = frame.drop(columns_to_drop, axis=1)

        print(frame_drop.columns)
        # creo le sequenze per la rete
        if cast_values:
            X, Y, ind_X, ind_Y = split_sequence(frame_drop.values.astype('float64'), self.seq_len_x, self.seq_len_y)
        else:
            X, Y, ind_X, ind_Y = split_sequence(frame_drop.values, self.seq_len_x, self.seq_len_y)

        ctfs = [list(frame_drop.columns).index(ctf) for ctf in columns_to_forecast]
        Y = Y[:, :, ctfs]
        # pagino il dataset e lo salvo nell'apposita cartella

        # rimuovo i valori che non conosco nell'outuput, questo serve a non provare a forecastare valori
        # i buchi della serie che sono stati riempiti dal fillna(0)
        # NB: i buchi nell'input sono accettati (X), sono quelli dell'output che creano problemi
        # per questo si rimuove da Y, di conseguenza, poi vengono tolti anche quegli input che portano
        # ad un forecast di uno 0
        if remove_not_known:
            # na_cols[ind_Y.reshape(ind_Y.shape[0],)] mi restituisce le colonne di na solo nei punti in cui esiste l'output y
            # scrivendo np.where(na_cols[ind_Y.reshape(ind_Y.shape[0],)] == True) prendo solo i valori in cui
            # l'output y era NA ed è stato refillato
            rp = np.where(na_cols[ind_Y.reshape(ind_Y.shape[0])] == True)[0]
            X = np.delete(X, rp, axis=0)
            Y = np.delete(Y, rp, axis=0)

        return X, Y

    @staticmethod
    def save_XY(X, Y, base_path, filename):
        filenames = []

        for idx, x in enumerate(X):
            fnx = f'{filename}_X_{idx}'
            fny = f'{filename}_Y_{idx}'
            filenames.append([fnx, fny])
            with open(base_path + fnx, 'wb') as output:
                pkl.dump(x, output)
            with open(base_path + fny, 'wb') as output:
                pkl.dump(Y[idx], output)
        np.save(f'{base_path}/{filename}_filenames.npy', filenames)
        print('salvato')

    """
        metodo per data augmentation:
        a partire dal dataset X,y generato dalla serie originale,
        vengono riprodotte num_replies copie dei punti precedenti 
        affetti da rumore 
    """

    @staticmethod
    def augment(X, Y, mean=0, variance=1.0, num_replies=5):
        sigma = variance ** 0.5
        new_X = []
        new_Y = []
        for n in range(num_replies):
            for idx, x in enumerate(X):
                x_gauss = np.random.normal(mean, sigma, (x.shape[0], x.shape[1]))
                y_gauss = np.random.normal(mean, sigma, (1, 1))
                new_X.append(x + x_gauss)
                new_Y.append(Y[idx] + y_gauss)
        return np.append(X, np.array(new_X), axis=0), np.append(Y, np.array(new_Y), axis=0)

    """
    target_col = 5
    unfold del dataset composto da (len(df['Rn_olb'])-30-1, 30, 12) elementi
    alla serie temporale di partenza
    """

    @staticmethod
    def get_ts_from_ds(X, y, target_col):
        rn = X[:, 0, target_col]
        rn = np.append(rn[:-1], X[-1, :, target_col])
        rn = np.append(rn, y[-1])
        return rn


def split_train_test_data(data_path, filename, columns, train_split=0.8):
    df = pd.read_csv(f'{data_path}/{filename}')
    df.columns = columns
    train_df, test_df = train_test_split(df, test_size=1 - train_split, shuffle=False)
    train_df.to_csv(f'{data_path}/train.csv', index=False)
    test_df.to_csv(f'{data_path}/test.csv', index=False)


def generate_dataset(data_path, filename, columns, seq_len_x=30, seq_len_y=1,
                     fill_na_type: FillnaTypes = FillnaTypes.MEAN,
                     remove_not_known=False):
    base_path = 'dataset/'
    encoders = f'{filename}-encoders/'
    scalers = f'{filename}-scalers/'

    if not os.path.exists(base_path):
        os.mkdir(base_path)
        print(f'{base_path} creata')

    if not os.path.exists(encoders):
        os.mkdir(encoders)
        print(f'{encoders} creata')

    if not os.path.exists(scalers):
        os.mkdir(scalers)
        print(f'{scalers} creata')

    dataset_generator = DatasetGenerator(data_path=data_path, columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y,
                                         encoders=encoders, scaler_path=scalers)
    X, y = dataset_generator.generate_XY(columns_to_scale=['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb'],
                                         columns_to_drop=['date', 'displacement (cm)',
                                                          'background seismicity', 'T_msa',
                                                          'Ru_msa', 'P_msa', 'Rn_msa'],
                                         columns_to_forecast=['Rn_olb'],
                                         fill_na_type=fill_na_type, remove_not_known=remove_not_known)
    # salvataggio trainin e test set
    dataset_generator.save_XY(X, y, base_path, filename)


if __name__ == '__main__':
    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']
    data_path = 'data'
    split_train_test_data(data_path=data_path, filename='olb_msa_full.csv', columns=columns)
    generate_dataset(data_path='data/train.csv', filename='train', columns=columns)
    generate_dataset(data_path='data/test.csv', filename='test', columns=columns, remove_not_known=True)
