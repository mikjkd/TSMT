from enum import Enum
from typing import List

import numpy as np


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


def fill_na_mean(df, target_columns: List):
    frame = df.copy()
    # per ogni colonna target sostituisco i valori nan
    for idx, c in enumerate(target_columns):
        # trovo le posizioni dei nan
        # alcuni valori non hanno nan, come ad esempio le date, per loro viene generata un'eccezione
        # e quindi si passa alla colonna successiva
        try:
            zero_pos = frame[np.isnan(frame[c].values)].index
            for zp in zero_pos:
                # primo valore precedente allo zero
                v0 = np.mean(frame[c])
                v1 = v0
                if zp > 0:
                    # prendo l'ultimo valore diverso da nan prima della posizione dello zero
                    try:
                        v0 = frame[:zp].values[~np.isnan(frame[c].values[:zp])][-1, idx]
                    except:
                        pass
                # primo valore successivo allo zero
                if zp < len(df):
                    # prendo il primo valore diverso da nan dopo la posizione dello zero
                    try:
                        v1 = frame[zp:].values[~np.isnan(frame[c].values[zp:])][0, idx]
                    except:
                        pass
                frame.at[zp, c] = (v0 + v1) / 2
        except Exception as e:
            # print(f'{c}: {e}')
            pass
    return frame


def fill_na(df, fill_na_type: FillnaTypes, columns):
    if fill_na_type == FillnaTypes.SIMPLE:
        df = df.fillna(0)
    elif fill_na_type == FillnaTypes.MEAN:
        df = fill_na_mean(df, columns)
    return df
