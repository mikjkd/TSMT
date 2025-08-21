import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def scale_preds(preds, scaler_path):
    # Implementazione della predizione
    scaler = joblib.load(scaler_path)
    scaled_preds = []
    for p in preds:
        scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
    # scaled_preds = [int(max(np.ceil(sp[0][0]), 0)) for sp in scaled_preds]
    return np.array(scaled_preds).reshape(len(scaled_preds))


def compare_scaled_values(regressor, generator, y_true):
    y_preds = regressor.model.predict(generator)
    scaled_y_true = scale_preds(y_true, scaler_path='scalers/Rn_olb_scaler.save')
    scaled_y_preds = scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')

    return zip(scaled_y_true, scaled_y_preds)


def plot_example_pred(generator, regressor):
    for i in range(20):
        n = np.random.randint(0, len(generator))
        val_true = generator[n]
        lstm_val_pred = regressor.model.predict(val_true[0])
        plt.plot(val_true[0][0, :, 4])
        plt.plot(len(val_true[0][0, :, 4]), val_true[1][0], 'x', label="true")
        plt.plot(len(val_true[0][0, :, 4]), lstm_val_pred[0], '-o', label="lstm_pred")
        plt.legend()
        plt.savefig(f'images/ex_{i}.png')


"""
    Il paper https://doi.org/10.1016/j.apradiso.2020.109239
    utilizza il coefficiente di Pearson per calcolare la correlazione tra il segnale misurato e l'allenato

    eval_pearsonsr presenta un'implementazione.
"""


def remove_outliers(scaled_y_true, scaled_y_preds, outliers_threshold):
    t = scaled_y_true.copy()
    p = scaled_y_preds.copy()
    out_thr = outliers_threshold
    wtr_y = np.where(t >= out_thr)[0]
    t = np.delete(t, wtr_y)
    p = np.delete(p, wtr_y)
    if len(p) == 0 or len(t) == 0:
        raise Exception('Threshold too low')
    return t, p


def eval_pearsonsr(y_preds, y_true, remove_outliers_p=False, outliers_threshold=120000,
                   scaler_path='scalers/Rn_olb_scaler.save'):
    y_true = y_true.reshape(-1)
    y_preds = y_preds.reshape(-1)
    if scaler_path is not None:
        scaled_y_true = scale_preds(y_true, scaler_path=scaler_path)
        scaled_y_preds = scale_preds(y_preds, scaler_path=scaler_path)
    else:
        scaled_y_true = y_true
        scaled_y_preds = y_preds
    if remove_outliers_p:
        scaled_y_true, scaled_y_preds = remove_outliers(scaled_y_true, scaled_y_preds, outliers_threshold)

    corr, _ = pearsonr(scaled_y_true, scaled_y_preds)
    print('Pearsons correlation: %.3f' % corr)
    return scaled_y_true, scaled_y_preds, corr


def eval(y_test, y_preds, scaler_path=None, remove_outliers=False):
    if scaler_path is None:
        scaler_path = 'scalers/Rn_olb_scaler.save'
    # non c'è bisogno di usare la classe corretta, basta usare la classe base

    _, __, ___, ____, pearsonval = eval_pearsonsr(y_preds, y_test, remove_outliers_p=remove_outliers,
                                                  scaler_path=scaler_path)
    # y_true = y_test.reshape(y_test.shape[0], )

    # scaled_preds = scale_preds(y_preds, scaler_path)
    # scaled_true = scale_preds(y_true, scaler_path)
    # plt.figure(figsize=(20, 6))
    # plt.plot(scaled_true, label='True')
    # plt.plot(scaled_preds,
    #          label='preds')
    # plt.legend()
    # plt.show()

    return pearsonval
