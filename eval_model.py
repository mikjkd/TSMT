import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def scale_preds(preds, scaler_path):
    scaler = joblib.load(scaler_path)
    scaled_preds = []
    for p in preds:
        scaled_preds.append(scaler.inverse_transform(p.reshape(-1, 1)))
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
The paper https://doi.org/10.1016/j.apradiso.2020.109239
uses the Pearson coefficient to calculate the correlation between the measured signal and the trained one.
eval_pearsonsr provides an implementation.
"""


def eval_pearsonsr(y_preds, y_true, scaler_path='scalers/Target_scaler.save'):
    y_true = y_true.reshape(-1)
    y_preds = y_preds.reshape(-1)
    scaled_y_true = scale_preds(y_true, scaler_path=scaler_path)
    scaled_y_preds = scale_preds(y_preds, scaler_path=scaler_path)

    corr, _ = pearsonr(scaled_y_true, scaled_y_preds)
    print('Pearsons correlation: %.3f' % corr)

    v_min = np.min([np.min(scaled_y_true), np.min(scaled_y_preds)])
    v_max = np.max([np.max(scaled_y_true), np.max(scaled_y_preds)])
    plt.plot(np.linspace(v_min, v_max), np.linspace(v_min, v_max))
    plt.scatter(scaled_y_preds, scaled_y_true)
    plt.show()
    return corr


def eval(y_test, y_preds, scaler_path=None):
    if scaler_path is None:
        scaler_path = 'scalers/Target_scaler.save'

    pearsonval = eval_pearsonsr(y_preds, y_test, scaler_path=scaler_path)
    y_true = y_test.reshape(y_test.shape[0], )

    scaled_preds = scale_preds(y_preds, scaler_path)
    scaled_true = scale_preds(y_true, scaler_path)
    plt.figure(figsize=(20, 6))
    plt.plot(scaled_true, label='True')
    plt.plot(scaled_preds,
             label='preds')
    plt.legend()
    plt.show()

    return pearsonval
