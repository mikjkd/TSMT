import matplotlib.pyplot as plt
import numpy as np

from model import LSTMRegressor, scale_preds


def compare_scaled_values(regressor, generator, y_true):
    y_preds = regressor.model.predict(generator)
    scaled_y_true = scale_preds(y_true, scaler_path='scalers/Rn_olb_scaler.save')
    scaled_y_preds = scale_preds(y_preds, scaler_path='scalers/Rn_olb_scaler.save')

    for z in zip(scaled_y_true, scaled_y_preds):
        print(f'true: {z[0]} ; pred: {z[1]}')


def plot_example_pred(generator, regressor):
    for i in range(20):
        n = np.random.randint(0, len(generator))
        val_true = generator[n]
        lstm_val_pred = regressor.model.predict(val_true[0])
        plt.plot(val_true[0][0, :, 4])
        plt.plot(len(val_true[0][0, :, 4]), val_true[1][0], 'x', label="true")
        plt.plot(len(val_true[0][0, :, 4]), lstm_val_pred[0], '-o', label="lstm_pred")
        plt.legend()
        plt.savefig(f'ex_{i}.png')
        plt.show()


def eval():
    lstm_model_name = 'lstm_model_no_zeros'

    lstm_regressor = LSTMRegressor(model_name=lstm_model_name, data_path='dataset/filenames.npy')

    lstm_regressor.load_model(f'saved_model/{lstm_model_name}.keras')

    data = lstm_regressor.load_data(shuffle=False)
    # divido i dati e creo i generators
    train_filenames, test_filenames = lstm_regressor.split_data(data)
    _, test_generator, __, ___ = lstm_regressor.generate_data(train_filenames, test_filenames)

    print('lstm eval')
    lstm_regressor.model.evaluate(test_generator)

    # devo estrarre le y dai generators
    y_true = []
    for y in test_generator:
        y_true.extend(y[1][:, 0, 0])
    y_true = np.array(y_true)

    # comparo i valori reali con i predetti
    # compare_scaled_values(lstm_regressor,test_generator, y_true)

    # disegno 30 step della serie e mostro la predizione
    # plot_example_pred(test_generator, lstm_regressor)


if __name__ == '__main__':
    eval()
