import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from data_generator import BaseDataset
from dataset import DatasetGenerator
from model import RegressorModel


def scale_preds(preds, scaler_path):
    # Implementazione della predizione
    scaler = joblib.load(scaler_path)
    scaled_preds = []
    for p in preds:
        scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
    #scaled_preds = [int(max(np.ceil(sp[0][0]), 0)) for sp in scaled_preds]
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


def eval_pearsonsr(y_preds, y_true, remove_outliers=False, scaler_path ='scalers/Rn_olb_scaler.save' ):
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
    corr, _ = pearsonr(y_true, y_preds)
    print('Pearsons correlation: %.3f' % corr)


    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds, scaled_y_true)
    plt.show()
    return corr


def eval(model_name):
    # non c'è bisogno di usare la classe corretta, basta usare la classe base
    regressor = RegressorModel(model_name=model_name)
    regressor.load_model(f'saved_model/{model_name}.x')
    data_path = f'dataset'

    dataset = BaseDataset(data_path=data_path)

    # carico i dati, li divido e creo i generators
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    # li carico già divisi, non serve più splittarli
    train_generator, test_generator, __, ___ = dataset.generate_data(train_filenames, test_filenames)
    regressor.model.evaluate(train_generator)
    y_preds = regressor.model.predict(test_generator)
    scaler_path = f'scalers'
    X_test, y_test = dataset.generator_to_Xy(test_generator)

    # eval_model.eval(model_name)
    lstm_y_preds = regressor.model.predict(test_generator)
    regressor.model.evaluate(test_generator)

    eval_pearsonsr(lstm_y_preds, y_test, remove_outliers=False)

    y_true = y_test.reshape(y_test.shape[0], )
    scaled_y_true = scale_preds(y_true, scaler_path=f'{scaler_path}/Rn_olb_scaler.save')
    scaled_y_preds = scale_preds(y_preds, scaler_path=f'{scaler_path}/Rn_olb_scaler.save')

    rn = DatasetGenerator.get_ts_from_ds(X_test, -2)
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(rn)
    plt.show()

    plt.plot(scaled_y_true, label='true')
    plt.plot(scaled_y_preds, label='preds')
    plt.legend()
    plt.show()


def eval_all_models(models):
    for idx, m in enumerate(models):
        print(f'\n\n\nModello numero {idx} : {m}')
        eval(m.split('.x')[0])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model = 'bd713292'
    eval(model)
    # eval()
