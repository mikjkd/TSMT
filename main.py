import json
import os.path

from keras.src.optimizers import Adam
from sklearn.metrics import mean_squared_error

import eval_model
from data_generator import BaseDataset
from dataset import DatasetGenerator, FillnaTypes
from model import ModelTrainer, generate_model_name
from models_repo.LSTMRegressor import LSTMRegressor

if __name__ == '__main__':
    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']
    columns_to_forecast = ['Rn_olb']
    columns_to_scale = ['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb']
    columns_to_drop = ['date', 'displacement (cm)', 'background seismicity', 'T_msa', 'Ru_msa', 'P_msa', 'Rn_msa']
    columns_to_filter = ['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb']
    seq_len_x = 30
    seq_len_y = 1
    data_path = 'data/olb_msa_full.csv'
    base_path = 'dataset/'
    encoders = 'encoders/'
    scalers = 'scalers/'
    batch_size = 64
    learning_rate = 0.001
    loss = 'mae'
    save = True
    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    df = dataset_generator.generate_frame()
    X, y = dataset_generator.generate_XY(df=df, columns_to_scale=['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb'],
                                         columns_to_drop=['date', 'displacement (cm)',
                                                          'background seismicity', 'T_msa',
                                                          'Ru_msa', 'P_msa', 'Rn_msa'],
                                         columns_to_forecast=['Rn_olb'],
                                         columns_to_filter=columns_to_filter,
                                         fill_na_type=FillnaTypes.MEAN, remove_not_known=False)
    # Creo train & test
    X_train, y_train = X[:int(len(X) * 0.8)], y[:int(len(y) * 0.8)]
    X_test, y_test = X[int(len(X) * 0.8):], y[int(len(y) * 0.8):]
    # Divido Train
    (X_train, y_train), (X_valid, y_valid) = BaseDataset.split_train_valid((X_train, y_train), shuffle=True)

    model_name = generate_model_name()
    regressor = LSTMRegressor(model_name=model_name)
    input_shape = (X.shape[1], X.shape[2])
    output_shape = 1
    regressor.generate_model(input_shape, output_shape)

    # train model

    trainer = ModelTrainer(batch_size=batch_size)
    epochs = 512
    trainer.run(
        model=regressor.model,
        model_name=regressor.model_name,
        train={'data': {'X': X_train, 'y': y_train}},
        valid={'data': {'X': X_valid, 'y': y_valid}},
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        epochs=epochs
    )

    lstm_y_preds = regressor.model.predict(X_test)
    regressor.model.evaluate(X_test, y_test)
    lstm_y_preds = lstm_y_preds.reshape(-1)

    mae = float(mean_squared_error(y_test.reshape(y_test.shape[0]), lstm_y_preds))

    pearsonsval = eval_model.eval(model_name, (X_test, y_test))

    if save:
        file_exists = os.path.isfile('performances.json')
        data = {
            'model_name': model_name, 'batch_size': batch_size,
            'epochs': epochs, 'metric_type': loss,
            'result': mae, 'pearson': pearsonsval,
            'numstep_in': seq_len_x,
            'numstep_out': seq_len_y,
            'model_features': {
                'model_type': 'LSTM',
                'model_description': regressor.description()
            },
            'dataset_features': {
                'filler_type': FillnaTypes.MEAN.value,
                'input_columns': columns_to_scale,
                'forecast_column': columns_to_forecast,
                'filtered_columns': columns_to_filter,
                'filter_type': 'Low'
            }
        }
        json_object = json.dumps(data, indent=4)
        fulljson = None
        if file_exists:
            with open("performances.json", "r") as jsonfile:
                fulljson = json.load(jsonfile)
            fulljson.append(data)
        else:
            fulljson = [data]
        with open("performances.json", "w+") as jsonfile:
            json.dump(fulljson, jsonfile)
