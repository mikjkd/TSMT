from enum import Enum
from typing import Union, Dict

import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class XYType(Enum):
    TRAIN = 'Train'
    TEST = 'Test'
    TRAINTEST = 'TrainTest'


class ScalerInfoTypes(Enum):
    INPUT = 'input'
    OUTPUT = 'output'


class ScalerTypes(Enum):
    MINMAX = 'minmax'
    STD = 'standard'


def minMaxScale(frame, pos):
    seq = frame[pos].values.astype('float64')
    scaler = MinMaxScaler()
    scaler = scaler.fit(seq.reshape(-1, 1))
    minmax = scaler.transform(seq.reshape(-1, 1))
    frame[pos] = minmax
    return scaler


def standardScale(frame, pos):
    seq = frame[pos].values.astype('float64')
    scaler = StandardScaler()
    scaler = scaler.fit(seq.reshape(-1, 1))
    std = scaler.transform(seq.reshape(-1, 1))
    frame[pos] = std
    return scaler


class Scaler:
    def __init__(self, io, name, function):
        self.io: Union[ScalerInfoTypes, None] = io
        self.name = name
        self.function = function


class ScalerColumns:
    def __init__(self, columns, scalers, scalerType: ScalerTypes = ScalerTypes.MINMAX):
        self.columns = columns
        self.scalers = scalers
        self.scalerType = scalerType


class ScalerInfo:
    def __init__(self):
        self.scalers: Dict[str, Scaler] = {}

    def load_from_map(self, scaler_map, xy_type: XYType):
        if xy_type == XYType.TRAIN:
            xy = 'training'
        else:
            xy = 'test'
        for cts in list(scaler_map.keys()):
            name = scaler_map[cts][xy]['name']
            io = scaler_map[cts][xy]['io']
            function = scaler_map[cts][xy]['function']
            scaler: Scaler = Scaler(name=name, io=io, function=function)
            self.scalers[cts] = scaler


def scale_from_columns(frame, scaler_path, columns_to_scale=None, scaler_names=None,
                       scalerType: ScalerTypes = ScalerTypes.MINMAX):
    if columns_to_scale is None:
        print('No columns to scale')
        return
    if scaler_names:
        # load provided scalers
        scalers = []
        for scaler_name in scaler_names:
            scalers.append(joblib.load(scaler_name))
        for idx, cts in enumerate(columns_to_scale):
            frame[cts] = scalers[idx].transform(
                frame[cts].values.reshape(-1, 1).astype('float64'))
    else:
        # define new scalers
        for cts in columns_to_scale:
            if scalerType == ScalerTypes.MINMAX:
                tmp_scaler = minMaxScale(frame, cts)
            else:
                tmp_scaler = standardScale(frame, cts)
            # save the scaler
            joblib.dump(tmp_scaler, scaler_path + cts + '_scaler.save')
    return frame


def scale_from_info(frame, scaler_path, scaler_info: ScalerInfo):
    columns = list(scaler_info.scalers.keys())
    for cts in columns:
        if scaler_info.scalers[cts].io == ScalerInfoTypes.INPUT:
            scaler = joblib.load(f"{scaler_path}{scaler_info.scalers[cts].name}_scaler.save")
            frame[cts] = scaler.transform(
                frame[cts].values.reshape(-1, 1).astype('float64'))
        elif scaler_info.scalers[cts].io == ScalerInfoTypes.OUTPUT:
            scaler = scaler_info.scalers[cts].function(frame, cts)
            # save the scaler
            joblib.dump(scaler, f"{scaler_path}{scaler_info.scalers[cts].name}_scaler.save")
        else:
            raise Exception('Wrong Scaler Info')
    return frame


def scale_df(frame, scaler_path, columns_to_scale):
    if columns_to_scale is None:
        raise Exception('Wrong Scale parameters')
    else:
        frame = frame.copy()

        if type(columns_to_scale) is ScalerInfo:
            frame = scale_from_info(frame, scaler_path, columns_to_scale)
        else:
            frame = scale_from_columns(frame, scaler_path, columns_to_scale.columns, columns_to_scale.scalers,
                                       columns_to_scale.scalerType)
        return frame
