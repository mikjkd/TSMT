import os
import pickle as pkl
from datetime import datetime, timedelta
from enum import Enum

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from .libV2 import minMaxScale, standardScale, split_sequence, fill_na_mean, IIR
from typing import Optional


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


class XYType(Enum):
    TRAIN = 'Train'
    TEST = 'Test'
    TRAINTEST = 'TrainTest'


"""
    DatasetGenerator Class provides a set of methods that process a DataFrame input
    generating X,y tensors for a supervised model training.
    !Those methods are specific for a TIME-SERIES domain.!
"""


class DatasetGenerator:
    def __init__(self, columns, data_path, encoders, scaler_path):
        self.columns = columns
        self.seq_len_x = 0
        self.seq_len_y = 0
        self.data_path = data_path
        self.encoders = encoders
        self.scaler_path = scaler_path
        # genXY parameters
        self.columns_to_scale = None
        self.columns_to_drop = None
        self.columns_to_forecast = None
        self.distributed = None
        self.filters = None
        self.fill_na_type = FillnaTypes.MEAN
        self.remove_not_known = False
        self.type = XYType.TEST
        self.train_test_split = None

    def load_XY(self):
        pass

    def generate_frame(self, start_date=None, end_date=None) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.columns = self.columns

        if start_date:
            df = df[df['date'] >= start_date]

        if end_date:
            df = df[df['date'] <= end_date]

        return df

    def set_dataset_values(self, seq_len_x, seq_len_y, columns_to_scale, columns_to_drop,
                           columns_to_forecast, distributed, filters, train_test_split,
                           fill_na_type=FillnaTypes.MEAN,
                           remove_not_known=False,
                           type=XYType.TEST):
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.columns_to_scale = columns_to_scale
        self.columns_to_drop = columns_to_drop
        self.columns_to_forecast = columns_to_forecast
        self.distributed = distributed
        self.filters = filters
        self.fill_na_type = fill_na_type
        self.remove_not_known = remove_not_known
        self.type = type
        self.train_test_split = train_test_split

    def __scale_df(self, frame, columns_to_scale=None, scaler_names=None,
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

    def __process_ds(self, frame, date_prediction_start=None, days_delta=7):
        frame = frame.copy()

        if date_prediction_start is not None:
            date_prediction_start = datetime.strptime(
                date_prediction_start, "%Y-%m-%d")
            date_ts_start = date_prediction_start - timedelta(days=days_delta)
            frame = frame[frame['date'] >= date_ts_start.strftime("%Y-%m-%d")]
            frame.reset_index(inplace=True, drop=True)
        return frame

    def __get_na_cols(self, df, columns_to_forecast):
        na_cols = np.isnan(df[columns_to_forecast].values)
        df = df.assign(na_cols=na_cols)
        return df, na_cols

    def __fill_na(self, df, fill_na_type: FillnaTypes):
        if fill_na_type == FillnaTypes.SIMPLE:
            df = df.fillna(0)
        elif fill_na_type == FillnaTypes.MEAN:
            df = fill_na_mean(df, self.columns)
        return df

    def __filter(self, df, colums_to_filter, type, order, cutoff, inplace):
        filters = [lfilter for i in range(len(colums_to_filter))]
        try:
            b, a = butter(order, cutoff, btype=type, analog=False)
            frame: pd.DataFrame = IIR(df, target_columns=colums_to_filter, filters=filters, a=a, b=b, inplace=inplace)
            if inplace is False:
                cnames = {f'filtered_{ctf}': f'{type}_filtered_{ctf}' for ctf in colums_to_filter}
                frame.rename(columns=cnames, inplace=True)
            return frame
        except:
            raise Exception('Wrong filter type')

    def __multi_filter(self, df: pd.DataFrame, filters) -> pd.DataFrame:
        frame = df.copy()
        for k in filters.keys():
            items = filters[k]['items']
            for i in items:
                ctf = [i['column']]
                order = i['parameters']['order']
                cutoff = i['parameters']['cutoff']
                frame = self.__filter(frame, colums_to_filter=ctf, type=k, order=order, cutoff=cutoff, inplace=False)
        return frame

    """generate_XY generates the X,y tensor needed for training/testing the model. It takes as input a DataFrame
    object -> the original data that we want to convert into tensors; The name of the columns that have to be scaled, 
    dropped, filtered (columns_to_scale, columns_to_drop, columns_to_filter); The values we want to learn ( 
    columns_to_forecast) are needed in order to create a target variable (y); Since the DataFrame often has missing 
    values (NA), a filling process is implemented, providing two ways (MEAN and SIMPLE), and it is possible to remove 
    filled values - through remove_not_know  - from the target (y) or use them for the forecast. Finally, this method 
    can be used in Training and Test mode: the principal difference is that when we use it in training mode, 
    the data are scaled and the generated scelers are saved, on the other side, during the test phase we use the same 
    scalers previously created, with TRAIN_TEST it is possible to call one time generate_XY and get (X_train,Y_train), 
    (X_test,Y_test)
    """

    def __generate_XY(self, df, columns_to_scale, columns_to_drop, columns_to_forecast, filters,
                      distributed,
                      cast_values=True,
                      remove_not_known=False,
                      scaler_type=ScalerTypes.MINMAX,
                      fill_na_type: FillnaTypes = FillnaTypes.SIMPLE,
                      type: XYType = XYType.TRAIN):
        # frame contiene le informazioni passate e viene processato dalla rete per creare delle predizioni
        # info frame contiene le informazioni che la rete sfrutta per migliorare le predizioni

        # copio le variabili altrimenti fuori dalla funzione risultano modificate e creano degli errori
        tcts = columns_to_scale.copy()
        tctd = columns_to_drop.copy()
        tct = columns_to_forecast.copy()
        # Adding the flag to the NAs columns
        df, na_cols = self.__get_na_cols(df, tct[0])
        ## filling na part -> A differenza del dataframe che può contenere valori vuoti (NaN), la sequenza (X,y) non può avere
        # valori NA
        # it is mandatory to fill na values in a (X,y) generation, it oly depends on the way
        # knowing the fact that we need to fill, a good way is to add a column to the df, saying that a specific value
        # has been filled with another value. That helps in post processing part
        df = self.__fill_na(df, fill_na_type=fill_na_type)
        # Filtro IIR
        inplace = False
        # frame = self.__filter(df=df, colums_to_filter=tctf, inplace=inplace)
        frame = self.__multi_filter(df=df, filters=filters)
        if inplace is False:
            # Siccome ho creato nuove colonne (filtered_...), se questi valori esistono nel set da scalare
            # allora aggiungo filtered_... in columns to scale.
            for k in filters.keys():
                items = filters[k]['items']
                for i in items:
                    if i["column"] in tcts:
                        tcts.append(f'{k}_filtered_{i["column"]}')
        # scalo le features e rimuovo quelle inutili
        # scalo a seconda del Training o Test mode
        if type == XYType.TRAIN:
            # se è training, genero gli scalers
            frame = self.__scale_df(frame, columns_to_scale=tcts,
                                    scalerType=scaler_type)
        elif type == XYType.TEST:
            # in caso di test, utilizzo gli scalers precedentemente creati
            scaler_names = [f'scalers/{cts}_scaler.save' for cts in tcts]
            frame = self.__scale_df(frame, columns_to_scale=tcts, scaler_names=scaler_names,
                                    scalerType=scaler_type)

        # creo le sequenze per la rete
        tctd.append('na_cols')
        target_columns_to_drop = tctd.copy()
        # se inplace è false, utilizzo anche le colonne filtrate come dataset
        # if inplace is False:
        #    target_columns_to_drop.extend(tctf)
        filtered_frame_drop = frame.drop(target_columns_to_drop, axis=1)
        target_frame_drop = frame.drop(tctd, axis=1)

        # print(filtered_frame_drop.columns, target_frame_drop.columns)

        if cast_values:
            # X, Y, ind_X, ind_Y = split_sequence(frame_drop.values.astype('float64'), self.seq_len_x, self.seq_len_y)
            X, _, ind_X, _ = split_sequence(filtered_frame_drop.values.astype('float64'), self.seq_len_x,
                                            self.seq_len_y)
            _, Y, _, ind_Y = split_sequence(target_frame_drop.values.astype('float64'), self.seq_len_x, self.seq_len_y,
                                            distributed=distributed)
        else:
            # X, Y, ind_X, ind_Y = split_sequence(frame_drop.values, self.seq_len_x, self.seq_len_y)
            X, _, ind_X, _ = split_sequence(filtered_frame_drop.values, self.seq_len_x,
                                            self.seq_len_y)
            _, Y, _, ind_Y = split_sequence(target_frame_drop.values, self.seq_len_x, self.seq_len_y,
                                            distributed=distributed)

        ctfs = [list(target_frame_drop.columns).index(ctf) for ctf in tct]
        if not distributed:
            Y = Y[:, :, ctfs]
        else:
            Y = Y[:, :, :, ctfs]
            Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

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

        # ripristino i valori iniziali
        return X, Y

    def generate_XY(self, df,
                    seq_len_x: Optional, seq_len_y: Optional,
                    columns_to_scale: Optional, columns_to_drop: Optional,
                    columns_to_forecast: Optional, filters: Optional,
                    cast_values=True,
                    remove_not_known=False,
                    scaler_type=ScalerTypes.MINMAX,
                    fill_na_type: FillnaTypes = FillnaTypes.SIMPLE,
                    type: XYType = XYType.TRAIN,
                    train_test_split=0.8, padding_size=0, distributed=False):
        # da fare ..
        # if seq_len_x is not None:
        #    self.seq_len_x = seq_len_x
        # if seq_len_y is not None:
        #    self.seq_len_y
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        if type == XYType.TRAIN or type == XYType.TEST:
            X, y = self.__generate_XY(df=df, columns_to_scale=columns_to_scale, columns_to_drop=columns_to_drop,
                                      columns_to_forecast=columns_to_forecast, filters=filters,
                                      cast_values=cast_values,
                                      remove_not_known=remove_not_known,
                                      scaler_type=scaler_type,
                                      fill_na_type=fill_na_type,
                                      type=type, distributed=distributed)
            return X, y
        elif type == XYType.TRAINTEST:
            train_df = df[:int(len(df) * train_test_split)]
            test_df = df[int(len(df) * train_test_split):]
            X_train, y_train = self.__generate_XY(df=train_df, columns_to_scale=columns_to_scale,
                                                  columns_to_drop=columns_to_drop,
                                                  columns_to_forecast=columns_to_forecast,
                                                  filters=filters,
                                                  cast_values=cast_values,
                                                  remove_not_known=remove_not_known,
                                                  scaler_type=scaler_type,
                                                  fill_na_type=fill_na_type,
                                                  type=XYType.TRAIN, distributed=distributed)
            # il padding_size in TRAINTEST viene utilizzato per una tecnica chiamata Alignment Buffer.
            # Questo buffer, composto dalle ultime 20 righe del training test posizionate prima del test
            # consente una migliore transizione della finestra, evitando un fenomeno che si chiama Boundary Effect.
            # Con questa finestra di buffer è possibile quindi avere una transizione più smooth dal training al test
            # con condizioni iniziali che hanno un senso nel caso in cui si divide in modo netto il dataset.
            if padding_size > 0:
                # Create overlap by taking the last 'overlap_size' rows from train_df
                overlap_df = train_df.iloc[-padding_size:].copy()
                test_df_with_overlap = pd.concat([overlap_df, test_df], ignore_index=True)
            else:
                test_df_with_overlap = test_df

            X_test, y_test = self.__generate_XY(df=test_df_with_overlap, columns_to_scale=columns_to_scale,
                                                columns_to_drop=columns_to_drop,
                                                columns_to_forecast=columns_to_forecast,
                                                filters=filters,
                                                cast_values=cast_values,
                                                remove_not_known=remove_not_known,
                                                scaler_type=scaler_type,
                                                fill_na_type=fill_na_type,
                                                type=XYType.TEST, distributed=distributed)
            return (X_train, y_train), (X_test, y_test)
        else:
            raise Exception("Wrong XY generation type")

    @staticmethod
    def save_XY(X, Y, base_path, filename):
        filenames = []

        for idx, x in enumerate(X):
            fnx = f'{filename}_X_{idx}'
            fny = f'{filename}_Y_{idx}'
            filenames.append([fnx, fny])
            with open(f'{base_path}/{fnx}', 'wb') as output:
                pkl.dump(x, output)
            with open(f'{base_path}/{fny}', 'wb') as output:
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
    def get_ts_from_ds(X, target_col):
        rn = X[0, :, target_col]
        rn = np.append(rn, X[1:, -1, target_col])
        return rn


def generate_dataset(save):
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
    columns_to_scale = ['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb']
    columns_to_drop = ['date', 'displacement (cm)',
                       'background seismicity', 'T_msa',
                       'Ru_msa', 'P_msa', 'Rn_msa']
    columns_to_forecast = ['Rn_olb']
    # filtering settings
    order = 1  # Order of the filter
    lp_cutoff = 0.3  # Cutoff frequency as a fraction of the Nyquist rate (0 to 1)
    hp_cutoff = 0.8
    filters = {
        'high': {
            'items': [
                {'column': 'Rn_olb', 'parameters': {'order': order, 'cutoff': hp_cutoff}}
            ],
        },
        'low': {
            'items': [
                {'column': 'Rn_olb', 'parameters': {'order': order, 'cutoff': lp_cutoff}}
            ],
        }
    }

    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    df = dataset_generator.generate_frame()
    X, y = dataset_generator.generate_XY(df=df, columns_to_scale=columns_to_scale,
                                         columns_to_drop=columns_to_drop,
                                         columns_to_forecast=columns_to_forecast,
                                         filters=filters,
                                         fill_na_type=FillnaTypes.MEAN, remove_not_known=False, type=XYType.TRAIN)
    train_test_split = 0.75
    # divisione train e test
    X_train, y_train = X[:int(len(X) * train_test_split)], y[:int(len(y) * train_test_split)]
    X_test, y_test = X[int(len(X) * train_test_split):], y[int(len(y) * train_test_split):]

    # salvataggio trainin e test set
    if save:
        dataset_generator.save_XY(X_train, y_train, base_path, 'train')
        dataset_generator.save_XY(X_test, y_test, base_path, 'test')


if __name__ == '__main__':
    generate_dataset(save=True)
