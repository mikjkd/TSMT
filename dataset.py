import pickle as pkl
from datetime import datetime, timedelta
from enum import Enum

import joblib
import numpy as np
import pandas as pd

from libV2 import minMaxScale, standardScale, split_sequence


class ScalerTypes(Enum):
    MINMAX = 'minmax'
    STD = 'standard'


class DatasetGenerator:
    def __init__(self, columns, seq_len_x, seq_len_y, data_path, encoders, scaler_path):
        self.columns = columns
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.data_path = data_path
        self.encoders = encoders
        self.scaler_path = scaler_path

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

    def generate_XY(self, columns_to_scale, columns_to_drop, columns_to_forecast, start_date=None, end_date=None,
                    save=True):
        # frame contiene le informazioni passate e viene processato dalla rete per creare delle predizioni
        # info frame contiene le informazioni che la rete sfrutta per migliorare le predizioni

        # merge dataset
        df = pd.read_csv(self.data_path)
        df.columns = self.columns

        if start_date:
            df = df[df['date'] >= start_date]

        if end_date:
            df = df[df['date'] <= end_date]

        df = df.fillna(0)

        # scalo le features e rimuovo quelle inutili

        frame = self.__scale_df(df, columns_to_scale=columns_to_scale, scalerType=ScalerTypes.MINMAX)
        frame_drop = frame.drop(columns_to_drop, axis=1)

        print(frame_drop.columns)
        # creo le sequenze per la rete
        X, Y = split_sequence(frame_drop.values.astype('float64'), self.seq_len_x, self.seq_len_y)
        ctfs = [list(frame_drop.columns).index(ctf) for ctf in columns_to_forecast]
        Y = Y[:, :, ctfs]
        # pagino il dataset e lo salvo nell'apposita cartella
        filenames = []
        if not save:
            return X, Y

        for idx, x in enumerate(X):
            fnx = f'X_{idx}'
            fny = f'Y_{idx}'
            filenames.append([fnx, fny])
            with open(base_path + fnx, 'wb') as output:
                pkl.dump(x, output)
            with open(base_path + fny, 'wb') as output:
                pkl.dump(Y[idx], output)
        np.save(f'{base_path}/filenames.npy', filenames)


if __name__ == '__main__':
    data_path = 'data/olb_msa_full.csv'
    base_path = 'dataset/'
    encoders = 'encoders/'
    scalers = 'scalers/'
    seq_len_x = 30
    seq_len_y = 1

    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']

    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    dataset_generator.generate_XY(columns_to_scale=['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb'],
                                  columns_to_drop=['date', 'displacement (cm)',
                                                   'background seismicity', 'T_msa',
                                                   'Ru_msa', 'P_msa', 'Rn_msa'],
                                  columns_to_forecast=['Rn_olb'],
                                  save=True)
    print('salvato')
