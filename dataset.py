import os
import pickle as pkl
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from .libV2 import split_sequence, fill_na_mean, IIR


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


class Constants(Enum):
    PADDING_SIZE = 'padding_size'
    SAVE = 'save'
    TRAINING_TYPE = 'training_type'
    TRAIN_TEST_SPLIT = 'train_test_split'
    FILTERS = 'filters'
    SCALERS_PATH = 'scalers'
    ENCODERS_PATH = 'encoders'
    BASE_PATH = 'base_path'
    DATA_PATH = 'data_path'
    SEQ_LEN_X = 'seq_len_x'
    SEQ_LEN_Y = 'seq_len_y'
    COLUMNS = 'columns'
    COLUMNS_TO_SCALE = 'columns_to_scale'
    COLUMNS_TO_DROP = 'columns_to_drop'
    COLUMNS_TO_FORECAST = 'columns_to_forecast'


class ScalerInfoTypes(Enum):
    INPUT = 'input'
    OUTPUT = 'output'


class Scaler:
    def __init__(self, io, name, function):
        self.io: Union[ScalerInfoTypes, None] = io
        self.name = name
        self.function = function


class ScalerInfo:
    """
        scaler_map = {
           'column_name':{
            'io': ScalerInfoTypes,
            'name': name,
            'function': Scaler function,
            }
           'column_name2':{
           'io': ScalerInfoType
            'name':name,
            'function':Scaler function
           }
        }
    """

    def __init__(self):
        self.scalers: Dict[str, Scaler] = {}

    def load_from_map(self, scaler_map):
        for cts in list(scaler_map.keys()):
            name = scaler_map[cts]['name']
            io = scaler_map[cts]['io']
            function = scaler_map[cts]['function']
            scaler: Scaler = Scaler(name=name, io=io, function=function)
            self.scalers[cts] = scaler


class Configuration:
    def __init__(self, xy_type, train_test_split, save, padding_size, seq_len_x, seq_len_y, columns_to_drop,
                 columns_to_scale, colums_to_forecast, filters):
        self.xy_type: XYType.TRAINTEST = xy_type
        self.train_test_split: float = train_test_split
        self.save: bool = save
        self.padding_size: int = padding_size
        self.seq_len_x: int = seq_len_x
        self.seq_len_y: int = seq_len_y
        self.columns_to_drop: list = columns_to_drop
        self.columns_to_scale = columns_to_scale
        self.columns_to_forecast: list = colums_to_forecast
        self.filters = filters


"""
    DatasetGenerator Class provides a set of methods that process a DataFrame input
    generating X,y tensors for a supervised model training.
    !Those methods are specific for a TIME-SERIES domain.!
"""


class Dataset:
    def __init__(self, columns, data_path, encoders, scaler_path):
        self.columns = columns
        self.seq_len_x = 0
        self.seq_len_y = 0
        self.data_path = data_path
        self.encoders = encoders
        self.scaler_path = scaler_path
        self.x_columns = None
        self.y_columns = None

    def generate_frame(self, start_date=None, end_date=None) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.columns = self.columns

        if start_date:
            df = df[df['date'] >= start_date]

        if end_date:
            df = df[df['date'] <= end_date]

        return df

    def scale_df(self, frame, scaler_info: ScalerInfo):
        frame = frame.copy()
        columns = list(scaler_info.scalers.keys())
        for cts in columns:
            if scaler_info.scalers[cts].io == ScalerInfoTypes.INPUT:
                scaler = joblib.load(f"{self.scaler_path}{scaler_info.scalers[cts].name}_scaler.save")
                frame[cts] = scaler.transform(
                    frame[cts].values.reshape(-1, 1).astype('float64'))
            elif scaler_info.scalers[cts].io == ScalerInfoTypes.OUTPUT:
                scaler = scaler_info.scalers[cts].function(frame, cts)
                # save the scaler
                joblib.dump(scaler, f"{self.scaler_path}{scaler_info.scalers[cts].name}_scaler.save")
            else:
                raise Exception('Wrong Scaler Info')
        return frame

    @staticmethod
    def process_ds(frame, date_prediction_start=None, days_delta=7):
        frame = frame.copy()

        if date_prediction_start is not None:
            date_prediction_start = datetime.strptime(
                date_prediction_start, "%Y-%m-%d")
            date_ts_start = date_prediction_start - timedelta(days=days_delta)
            frame = frame[frame['date'] >= date_ts_start.strftime("%Y-%m-%d")]
            frame.reset_index(inplace=True, drop=True)
        return frame

    @staticmethod
    def get_na_cols(df, columns_to_forecast):
        na_cols = np.isnan(df[columns_to_forecast].values)
        new_df = df.copy()
        for idx, c in enumerate(columns_to_forecast):
            new_df[f'na_{c}'] = na_cols[:, idx]
        return new_df, na_cols

    def fill_na(self, df, fill_na_type: FillnaTypes):
        if fill_na_type == FillnaTypes.SIMPLE:
            df = df.fillna(0)
        elif fill_na_type == FillnaTypes.MEAN:
            df = fill_na_mean(df, self.columns)
        return df

    @staticmethod
    def __filter(df, colums_to_filter, filter_type, order, cutoff, inplace):
        filters = [lfilter for _ in range(len(colums_to_filter))]
        try:
            b, a = butter(order, cutoff, btype=filter_type, analog=False)
            frame: pd.DataFrame = IIR(df, target_columns=colums_to_filter, filters=filters, a=a, b=b, inplace=inplace)
            if inplace is False:
                cnames = {f'filtered_{ctf}': f'{filter_type}_filtered_{ctf}' for ctf in colums_to_filter}
                frame.rename(columns=cnames, inplace=True)
            return frame
        except:
            raise Exception('Wrong filter type')

    def multi_filter(self, df: pd.DataFrame, filters) -> pd.DataFrame:
        frame = df.copy()
        for k in filters.keys():
            items = filters[k]['items']
            for i in items:
                ctf = [i['column']]
                order = i['parameters']['order']
                cutoff = i['parameters']['cutoff']
                frame = self.__filter(frame, colums_to_filter=ctf, filter_type=k, order=order, cutoff=cutoff,
                                      inplace=False)
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
                      fill_na_type: FillnaTypes = FillnaTypes.SIMPLE):
        tcts = columns_to_scale.copy()
        tctd = columns_to_drop.copy()
        tct = columns_to_forecast.copy()
        # Adding the flag to the NAs columns
        df, na_cols = self.get_na_cols(df, tct)

        # it is mandatory to fill na values in a (X,y) generation, it oly depends on the way
        # knowing the fact that we need to fill, a good way is to add a column to the df, saying that a specific value
        # has been filled with another value. That helps in post processing part
        df = self.fill_na(df, fill_na_type=fill_na_type)
        inplace = False

        frame = self.multi_filter(df=df, filters=filters)
        if inplace is False:
            # update columns to scale according to the new filtered columns
            for k in filters.keys():
                items = filters[k]['items']
                for i in items:
                    if i["column"] in list(tcts.keys()):
                        tcts[f'{k}_filtered_{i["column"]}'] = tcts[i["column"]]

        scaler_info: ScalerInfo = ScalerInfo()
        scaler_info.load_from_map(tcts)
        frame = self.scale_df(frame, scaler_info=scaler_info)

        tctd.extend([f'na_{c}' for c in tct])
        frame_drop = frame.drop(tctd, axis=1)
        # I save the new X,y column names
        self.x_columns = frame_drop.columns

        if cast_values:
            X, _, ind_X, _ = split_sequence(frame_drop.values.astype('float64'), self.seq_len_x,
                                            self.seq_len_y)
            _, Y, _, ind_Y = split_sequence(frame_drop.values.astype('float64'), self.seq_len_x, self.seq_len_y,
                                            distributed=distributed)
        else:
            X, _, ind_X, _ = split_sequence(frame_drop.values, self.seq_len_x,
                                            self.seq_len_y)
            _, Y, _, ind_Y = split_sequence(frame_drop.values, self.seq_len_x, self.seq_len_y,
                                            distributed=distributed)

        ctfs = [list(frame_drop.columns).index(ctf) for ctf in tct]
        if self.seq_len_y > 0:
            Y = Y[:, :, ctfs]

        self.y_columns = tct

        # I remove the values I don’t know in the output; this is to avoid trying to forecast values
        # that are gaps in the series filled with fillna(0)
        # Note: gaps in the input (X) are accepted; it’s the gaps in the output that cause problems
        # For this reason, values are removed from Y, and as a consequence, the input values
        # leading to a forecast of 0 are also removed

        if remove_not_known:
            rp = np.where(na_cols[ind_Y.reshape(ind_Y.shape[0])] is True)[0]
            X = np.delete(X, rp, axis=0)
            Y = np.delete(Y, rp, axis=0)

        print("Done!")
        return X, Y

    def generate_XY(self, df,
                    seq_len_x: Optional, seq_len_y: Optional,
                    columns_to_scale: Optional, columns_to_drop: Optional,
                    columns_to_forecast: Optional, filters: Optional,
                    cast_values=True,
                    remove_not_known=False,
                    fill_na_type: FillnaTypes = FillnaTypes.SIMPLE,
                    xy_type: XYType = XYType.TRAIN,
                    train_test_split=0.8, padding_size=0, distributed=False):
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        if xy_type == XYType.TRAIN or xy_type == XYType.TEST:
            for cts in list(columns_to_scale.keys()):
                columns_to_scale[cts][
                    'io'] = ScalerInfoTypes.OUTPUT if xy_type == XYType.TRAIN else ScalerInfoTypes.INPUT
            X, y = self.__generate_XY(df=df, columns_to_scale=columns_to_scale, columns_to_drop=columns_to_drop,
                                      columns_to_forecast=columns_to_forecast, filters=filters,
                                      cast_values=cast_values,
                                      remove_not_known=remove_not_known,
                                      fill_na_type=fill_na_type,
                                      distributed=distributed)
            return X, y
        elif xy_type == XYType.TRAINTEST:
            train_df = df[:int(len(df) * train_test_split)]
            test_df = df[int(len(df) * train_test_split):]
            for cts in list(columns_to_scale.keys()):
                columns_to_scale[cts]['io'] = ScalerInfoTypes.OUTPUT
            X_train, y_train = self.__generate_XY(df=df, columns_to_scale=columns_to_scale,
                                                  columns_to_drop=columns_to_drop,
                                                  columns_to_forecast=columns_to_forecast, filters=filters,
                                                  cast_values=cast_values,
                                                  remove_not_known=remove_not_known,
                                                  fill_na_type=fill_na_type,
                                                  distributed=distributed)
            # The padding_size in TRAINTEST is used for a technique called Alignment Buffer.
            # This buffer, consisting of the last 20 rows of the training set positioned before the test set,
            # allows for a smoother transition of the window, avoiding a phenomenon called the Boundary Effect.
            # With this buffer window, it is possible to achieve a smoother transition from training to test,
            # with meaningful initial conditions in cases where the dataset is split sharply.
            if padding_size > 0:
                # Create overlap by taking the last 'overlap_size' rows from train_df
                overlap_df = train_df.iloc[-padding_size:].copy()
                test_df_with_overlap = pd.concat([overlap_df, test_df], ignore_index=True)
            else:
                test_df_with_overlap = test_df

            for cts in list(columns_to_scale.keys()):
                columns_to_scale[cts]['io'] = ScalerInfoTypes.INPUT
            X_test, y_test = self.__generate_XY(df=test_df_with_overlap, columns_to_scale=columns_to_scale,
                                                columns_to_drop=columns_to_drop,
                                                columns_to_forecast=columns_to_forecast,
                                                filters=filters,
                                                cast_values=cast_values,
                                                remove_not_known=remove_not_known,
                                                fill_na_type=fill_na_type,
                                                distributed=distributed)
            return (X_train, y_train), (X_test, y_test)
        else:
            # training type is none, it means that there are no constraint on the kind of scaling
            X, y = self.__generate_XY(df=df, columns_to_scale=columns_to_scale, columns_to_drop=columns_to_drop,
                                      columns_to_forecast=columns_to_forecast, filters=filters,
                                      cast_values=cast_values,
                                      remove_not_known=remove_not_known,
                                      fill_na_type=fill_na_type,
                                      distributed=distributed)

            return X, y

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
        print('saved')

    @staticmethod
    def get_ts_from_ds(X, target_col):
        rn = X[0, :, target_col]
        rn = np.append(rn, X[1:, -1, target_col])
        return rn


def generate_dataset(configuration):
    data_path = configuration[Constants.DATA_PATH]  # data/dataset.csv
    base_path = configuration[Constants.BASE_PATH]  # dataset/'
    encoders = configuration[Constants.ENCODERS_PATH]  # 'encoders/'
    scalers = configuration[Constants.SCALERS_PATH]  # 'scalers/'

    if not os.path.exists(base_path):
        os.mkdir(base_path)
        print(f'{base_path} creata')

    if not os.path.exists(encoders):
        os.mkdir(encoders)
        print(f'{encoders} creata')

    if not os.path.exists(scalers):
        os.mkdir(scalers)
        print(f'{scalers} creata')

    seq_len_x = configuration[Constants.SEQ_LEN_X]  # 30
    seq_len_y = configuration[Constants.SEQ_LEN_Y]  # 1

    columns = configuration[Constants.COLUMNS]
    columns_to_scale = configuration[Constants.COLUMNS_TO_SCALE]
    columns_to_drop = configuration[Constants.COLUMNS_TO_DROP]
    columns_to_forecast = configuration[Constants.COLUMNS_TO_FORECAST]
    # filtering settings
    filters = configuration[Constants.FILTERS]

    dataset = Dataset(columns=columns, data_path=data_path,
                      encoders=encoders, scaler_path=scalers)

    df = dataset.generate_frame()
    training_type = configuration[Constants.TRAINING_TYPE]  # XYType.TRAIN
    if training_type == XYType.TRAIN or training_type == XYType.TEST:
        X, y = dataset.generate_XY(df=df, seq_len_x=seq_len_x, seq_len_y=seq_len_y,
                                   columns_to_scale=columns_to_scale,
                                   columns_to_drop=columns_to_drop,
                                   columns_to_forecast=columns_to_forecast,
                                   filters=filters,
                                   fill_na_type=FillnaTypes.MEAN, remove_not_known=False,
                                   xy_type=training_type)
        if configuration[Constants.SAVE]:
            dataset.save_XY(X, y,
                            base_path,
                            'train' if
                            configuration[Constants.TRAINING_TYPE] == XYType.TRAIN
                            else 'test')
    else:
        (X_train, y_train), (X_test, y_test) = dataset.generate_XY(df=df, seq_len_x=seq_len_x,
                                                                   seq_len_y=seq_len_y,
                                                                   columns_to_scale=columns_to_scale,
                                                                   columns_to_drop=columns_to_drop,
                                                                   columns_to_forecast=columns_to_forecast,
                                                                   filters=filters,
                                                                   fill_na_type=FillnaTypes.MEAN,
                                                                   train_test_split=configuration[
                                                                       Constants.TRAIN_TEST_SPLIT],
                                                                   remove_not_known=False,
                                                                   padding_size=configuration[Constants.PADDING_SIZE],
                                                                   xy_type=training_type)
        if configuration[Constants.SAVE]:
            dataset.save_XY(X_train, y_train, base_path, 'train')
            dataset.save_XY(X_test, y_test, base_path, 'test')
