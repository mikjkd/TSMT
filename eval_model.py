import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from data_generator import BaseDataset, CustomOpsGenerator
from model import RegressorModel


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
        plt.show()


"""
    Il paper https://doi.org/10.1016/j.apradiso.2020.109239
    utilizza il coefficiente di Pearson per calcolare la correlazione tra il segnale misurato e l'allenato
    
    eval_pearsonsr presenta un'implementazione.
"""


def eval_pearsonsr(y_preds, y_true, remove_outliers=False, scaler_path='scalers/Rn_olb_scaler.save'):
    y_true = y_true.reshape(-1)
    y_preds = y_preds.reshape(-1)
    scaled_y_true = scale_preds(y_true, scaler_path=scaler_path)
    scaled_y_preds = scale_preds(y_preds, scaler_path=scaler_path)
    """if remove_outliers:
        out_thr = 40000
        wtr_y = np.where(scaled_y_true >= out_thr)[0]
        scaled_y_true = np.delete(scaled_y_true, wtr_y)
        scaled_y_preds = np.delete(scaled_y_preds, wtr_y)
        # for z in zip(scaled_y_true, scaled_y_preds):
        #    print(f'true: {z[0]} ; pred: {z[1]}')
        wtr_x = np.where(scaled_y_preds >= out_thr)[0]
        scaled_y_true = np.delete(scaled_y_true, wtr_x)
        scaled_y_preds = np.delete(scaled_y_preds, wtr_x)
    """
    corr, _ = pearsonr(scaled_y_true, scaled_y_preds)
    print('Pearsons correlation: %.3f' % corr)

    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds, scaled_y_true)
    plt.show()
    return corr


def eval(model_name, data, scaler_path=None):
    if scaler_path is None:
        scaler_path = 'scalers/Rn_olb_scaler.save'
    # non c'è bisogno di usare la classe corretta, basta usare la classe base
    regressor = RegressorModel(model_name=model_name)
    regressor.load_model(f'saved_model/{model_name}.x')
    if type(data) is CustomOpsGenerator:
        X_test, y_test = BaseDataset.generator_to_Xy(data)
        # eval_model.eval(model_name)
        y_preds = regressor.model.predict(data)
        regressor.model.evaluate(data)

    elif type(data) is tuple:
        y_preds = regressor.model.predict(data[0])
        regressor.model.evaluate(data[0], data[1])
        X_test, y_test = data[0], data[1]
    else:
        raise Exception('Wrong data type')
    pearsonval = eval_pearsonsr(y_preds, y_test, remove_outliers=False, scaler_path=scaler_path)
    y_true = y_test.reshape(y_test.shape[0], )
    # plt.figure(figsize=(20, 6), dpi=80)
    # plt.plot(y_true, label='true')
    # plt.plot(y_preds, label='preds')
    # plt.legend()
    # plt.show()

    scaled_preds = scale_preds(y_preds, scaler_path)
    scaled_true = scale_preds(y_true, scaler_path)
    plt.figure(figsize=(20, 6))
    plt.plot(scaled_true, label='True')
    plt.plot(scaled_preds,
             label='preds')
    plt.legend()
    plt.show()

    return pearsonval
