from typing import List

from scipy.signal import lfilter, butter
import pandas as pd


def IIR_highpass(y_prec, x_curr, x_prec, a: float = 0.8):
    y_curr = a * y_prec + (x_curr - x_prec) * ((1 + a) / 2)
    return y_curr


def apply_filter(x, a, b, filter):
    return filter(b, a, x)


"""
IIR spike filter.  
The filter can only be applied when there are no more NaN values, so an imputing strategy must be used first.
The `filters` parameter must be a list of filters, like the following:  
    filters = [IIR_highpass, ...]

"""


def IIR(df: pd.DataFrame, target_columns: List, filters: List, a, b, inplace=False) -> pd.DataFrame:
    frame = df.copy()
    for idx, c in enumerate(target_columns):
        try:
            if inplace:
                name = c
            else:
                name = f'filtered_{c}'
            x = frame[c].values
            frame[name] = apply_filter(x, a, b, filters[idx])
        except Exception as e:
            print(e)
            raise
    return frame


def filter(df, colums_to_filter, filter_type, order, cutoff, inplace):
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


def multi_filter(df: pd.DataFrame, filters) -> pd.DataFrame:
    frame = df.copy()
    for k in filters.keys():
        items = filters[k]['items']
        for i in items:
            ctf = [i['column']]
            order = i['parameters']['order']
            cutoff = i['parameters']['cutoff']
            frame = filter(frame, colums_to_filter=ctf, filter_type=k, order=order, cutoff=cutoff,
                           inplace=False)
    return frame
