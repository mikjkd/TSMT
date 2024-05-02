import joblib
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
from scipy.stats import pearsonr

from dataset import DatasetGenerator, FillnaTypes
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


def eval_pearsonsr(y_preds, y_true, remove_outliers=False):
    y_true = y_true.reshape(y_true.shape[0], )
    scaled_y_true = scale_preds(y_true, scaler_path='scalers/Rn_olb_scaler.save')
    scaled_y_preds = scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')
    if remove_outliers:
        wtr_y = np.where(scaled_y_true >= 100000)[0]
        scaled_y_true = np.delete(scaled_y_true, wtr_y)
        scaled_y_preds = np.delete(scaled_y_preds, wtr_y)
        # for z in zip(scaled_y_true, scaled_y_preds):
        #    print(f'true: {z[0]} ; pred: {z[1]}')
        wtr_x = np.where(scaled_y_preds >= 100000)[0]
        scaled_y_true = np.delete(scaled_y_true, wtr_x)
        scaled_y_preds = np.delete(scaled_y_preds, wtr_x)

    corr, _ = pearsonr(y_true, y_preds.reshape(-1))
    print('Pearsons correlation: %.3f' % corr)
    scaled_corr, _ = pearsonr(scaled_y_true, scaled_y_preds)
    print('Pearsons correlation on scaled vals: %.3f' % scaled_corr)

    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds, scaled_y_true)
    plt.show()


def eval(lstm_model_name='lstm_mae_model_with_valid_bs32'):
    lstm_regressor = LSTMRegressor(model_name=lstm_model_name)
    lstm_regressor.load_model(f'saved_model/{lstm_model_name}.keras')

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

    X, y = dataset_generator.generate_XY(columns_to_scale=['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb'],
                                         columns_to_drop=['date', 'displacement (cm)',
                                                          'background seismicity', 'T_msa',
                                                          'Ru_msa', 'P_msa', 'Rn_msa'],
                                         columns_to_forecast=['Rn_olb'], cast_values=False,
                                         fill_na_type=FillnaTypes.SIMPLE, remove_not_known=True)

    X_train, y_train = X[:floor(len(X) * 0.8)], y[:floor(len(y) * 0.8)]
    X_test, y_test = X[ceil(len(X) * 0.8):], y[ceil(len(y) * 0.8):]

    y_preds = lstm_regressor.model.predict(X_test)

    scaler = joblib.load('scalers/Rn_olb_scaler.save')

    diffs = []
    scaled_y_true = []
    scaled_y_preds = []
    for v in zip(scale_preds(y_test.reshape(y_test.shape[0]), scaler_path='scalers/Rn_olb_scaler.save'),
                 scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')):
        diffs.append(np.abs(v[0] - v[1]))
        scaled_y_true.append(v[0])
        scaled_y_preds.append(v[1])

    diffs = np.array(diffs)
    scaled_y_preds = np.array(scaled_y_preds)
    scaled_y_true = np.array(scaled_y_true)

    print(np.mean(diffs), np.min(diffs), np.max(diffs))

    eval_pearsonsr(y_preds, y_test)

    rn = DatasetGenerator.get_ts_from_ds(X_test, y_test, -1)
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
        eval(m.split('.keras')[0])


if __name__ == '__main__':
    # models = os.listdir((os.getcwd()) + '/saved_model')
    model = 'lstm_mse_model_with_valid_bs64_Adam_lr0.001'
    eval(model)
    # eval_all_models(models)
    # eval()
