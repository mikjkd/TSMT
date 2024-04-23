import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from data_generator import BaseDataset
from dataset import DatasetGenerator
from model import LSTMRegressor


def scale_preds(preds, scaler_path):
    # Implementazione della predizione
    scaler = joblib.load(scaler_path)
    scaled_preds = []
    for p in preds:
        scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
    scaled_preds = [int(max(np.ceil(sp[0][0]), 0)) for sp in scaled_preds]
    return np.array(scaled_preds)


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


def eval_pearsonsr(regressor, generator, y_true):
    y_preds = regressor.model.predict(generator)
    scaled_y_true = scale_preds(y_true, scaler_path='scalers/Rn_olb_scaler.save')
    scaled_y_preds = scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')

    # wtr = np.where(scaled_y_true > 120000)[0]
    # scaled_y_true = np.delete(scaled_y_true, wtr)
    # scaled_y_preds = np.delete(scaled_y_preds, wtr)
    for z in zip(scaled_y_true, scaled_y_preds):
        print(f'true: {z[0]} ; pred: {z[1]}')

    corr, _ = pearsonr(y_true, y_preds.reshape(-1))
    print('Pearsons correlation: %.3f' % corr)
    scaled_corr, _ = pearsonr(scaled_y_true, scaled_y_preds)
    print('Pearsons correlation on scaled vals: %.3f' % scaled_corr)

    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds, scaled_y_true)
    plt.show()


def eval():
    lstm_model_name = 'lstm_mae_model_with_valid' #'lstm_mae_model'
    lstm_regressor = LSTMRegressor(model_name=lstm_model_name)
    lstm_regressor.load_model(f'saved_model/{lstm_model_name}.keras')

    dataset = BaseDataset(data_path='dataset')
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    # divido i dati e creo i generators
    train_generator, test_generator, __, ___ = dataset.generate_data(train_filenames, test_filenames)

    print('lstm eval')
    lstm_regressor.model.evaluate(test_generator)

    # devo estrarre le y dai generators
    y_true = []
    for y in test_generator:
        y_true.extend(y[1][:, 0, 0])
    y_true = np.array(y_true)

    y_preds = lstm_regressor.model.predict(test_generator)

    scaler = joblib.load('scalers/Rn_olb_scaler.save')

    # comparo i valori reali con i predetti

    vals = compare_scaled_values(lstm_regressor, test_generator, y_true)

    diffs = []
    scaled_y_true = []
    scaled_y_preds = []
    for v in vals:
        diffs.append(np.abs(v[0] - v[1]))
        scaled_y_true.append(v[0])
        scaled_y_preds.append(v[1])

    diffs = np.array(diffs)
    scaled_y_preds = np.array(scaled_y_preds)
    scaled_y_true = np.array(scaled_y_true)

    print(np.mean(diffs), np.min(diffs), np.max(diffs))

    mean_p = np.where(np.abs(scaled_y_true - scaled_y_preds) >= np.mean(diffs))
    inv_mean_p = np.where(np.abs(scaled_y_true - scaled_y_preds) < np.mean(diffs))

    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds[mean_p], scaled_y_true[mean_p], color='red')
    plt.scatter(scaled_y_preds[inv_mean_p], scaled_y_true[inv_mean_p], color='green')
    plt.show()

    x_vals = list(range(len(scaled_y_preds)))
    print(len(scaled_y_preds[mean_p]), len(scaled_y_preds[inv_mean_p]))

    thr_vals = scaled_y_true.copy().astype(float)
    thr_vals[inv_mean_p] = np.nan

    ok_vals = scaled_y_true.copy().astype(float)
    ok_vals[mean_p] = np.nan
    plt.plot(x_vals, scaled_y_true, color='black')
    # plt.scatter(x_vals, ok_vals, marker='x', color='green')
    plt.scatter(x_vals, thr_vals, marker='x', color='red')
    plt.show()

    # disegno 30 step della serie e mostro la predizione
    # plot_example_pred(test_generator, lstm_regressor)

    eval_pearsonsr(lstm_regressor, test_generator, y_true)

    # plot real vs forecast
    X = []
    y = []
    for x in test_generator:
        X.extend(x[0])
        y.extend(x[1])

    rn = DatasetGenerator.get_ts_from_ds(np.array(X), np.array(y), target_col=-1)
    plt.figure(figsize=(20,6), dpi=80)
    plt.plot(rn)
    plt.show()

    # plot train set
    X = []
    y = []
    for x in train_generator:
        X.extend(x[0])
        y.extend(x[1])

    rn = DatasetGenerator.get_ts_from_ds(np.array(X), np.array(y), target_col=-1)
    plt.figure(figsize=(20,6), dpi=80)
    plt.plot(rn)
    plt.show()

    plt.plot(scaled_y_true, label='true')
    plt.plot(scaled_y_preds, label='preds')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    eval()
