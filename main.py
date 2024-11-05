import json
import os.path

from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_error

import eval_model
from data_generator import BaseDataset
from dataset import DatasetGenerator, FillnaTypes, XYType
from model import ModelTrainer, generate_model_name
from models_repo.LSTMRegressor import LSTMRegressor, TDLSTMRegressor


def save_results(model_name, batch_size, epochs, loss, mae, pearsonsval, seq_len_x, seq_len_y,
                 regressor, timedistributed, columns_to_scale, columns_to_forecast, filters,
                 train_test_split, padding_size):
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
            'time_distributed': timedistributed,
            'filler_type': FillnaTypes.MEAN.value,
            'input_columns': columns_to_scale,
            'forecast_column': columns_to_forecast,
            'filters_settings': filters,
            'filter_type': 'Low',
            'train_test_split': train_test_split,
            'padding_size': padding_size
        }
    }
    if file_exists:
        with open("performances.json", "r") as jsonfile:
            fulljson = json.load(jsonfile)
        fulljson.append(data)
    else:
        fulljson = [data]
    with open("performances.json", "w+") as jsonfile:
        json.dump(fulljson, jsonfile)


def main():
    columns = ['date', 'RSAM', 'T_olb',
               'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']
    columns_to_forecast = ['Rn_olb']
    columns_to_scale = ['RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb']
    columns_to_drop = ['date', 'displacement (cm)', 'background seismicity', 'T_msa', 'Ru_msa', 'P_msa', 'Rn_msa']
    # filtering settings
    order = 1  # Order of the filter
    lp_cutoff = 0.3  # Cutoff frequency as a fraction of the Nyquist rate (0 to 1)
    hp_cutoff = 0.8
    filters = {
        'high': {
            'items': [
                {'column': 'Rn_olb', 'parameters': {'order': order, 'cutoff': hp_cutoff}}
            ],
        },
        'low': {
            'items': [
                # {'column': 'RSAM', 'parameters': {'order': order, 'cutoff': lp_cutoff}},
                # {'column': 'T_olb', 'parameters': {'order': order, 'cutoff': lp_cutoff}},
                # {'column': 'Ru_olb', 'parameters': {'order': order, 'cutoff': lp_cutoff}},
                # {'column': 'P_olb', 'parameters': {'order': order, 'cutoff': lp_cutoff}},
                {'column': 'Rn_olb', 'parameters': {'order': order, 'cutoff': lp_cutoff}}
            ],
        }
    }
    seq_len_x = 30
    seq_len_y = 1
    data_path = 'data/olb_msa_full.csv'
    base_path = 'dataset/'
    encoders = 'encoders/'
    scalers = 'scalers/'
    batch_size = 32
    learning_rate = 0.001
    train_test_split = 0.75
    loss = 'mae'
    save = True
    padding_size = 30
    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    df = dataset_generator.generate_frame()
    timedistributed = False
    (X_train, y_train), (X_test, y_test) = dataset_generator.generate_XY(df=df,
                                                                         columns_to_scale=columns_to_scale,
                                                                         columns_to_drop=columns_to_drop,
                                                                         columns_to_forecast=columns_to_forecast,
                                                                         distributed=timedistributed,
                                                                         filters=filters,
                                                                         fill_na_type=FillnaTypes.MEAN,
                                                                         remove_not_known=False,
                                                                         type=XYType.TRAINTEST,
                                                                         train_test_split=train_test_split,
                                                                         padding_size=padding_size)
    # Divido Train
    (X_train, y_train), (X_valid, y_valid) = BaseDataset.split_train_valid((X_train, y_train), shuffle=True)
    model_name = generate_model_name()
    if timedistributed:
        regressor = TDLSTMRegressor(model_name=model_name)
    else:
        regressor = LSTMRegressor(model_name=model_name)
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = 1
    regressor.generate_model(input_shape, output_shape)

    # train model

    trainer = ModelTrainer(batch_size=batch_size)
    epochs = 1024
    trainer.run(
        model=regressor.model,
        model_name=regressor.model_name,
        train={'data': {'X': X_train, 'y': y_train}},
        valid={'data': {'X': X_valid, 'y': y_valid}},
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        epochs=epochs
    )
    # loading the best model
    regressor.load_model(f'saved_model/{model_name}.x')
    # Adjust test set by excluding the overlap
    X_test_eval = X_test  # [padding_size:]
    y_test_eval = y_test  # [padding_size:]

    # Model predictions
    lstm_y_preds = regressor.model.predict(X_test_eval)
    pearsonsval = eval_model.eval(y_test[:, -1], lstm_y_preds[:, -1])
    # Calculate Mean Absolute Error (MAE)
    mae = float(mean_absolute_error(y_test_eval.reshape(y_test_eval.shape[0]), lstm_y_preds))

    # Calculate Pearson's correlation => i have a filtered X

    if save:
        save_results(model_name, batch_size, epochs, loss, mae, pearsonsval, seq_len_x, seq_len_y,
                     regressor, timedistributed, columns_to_scale, columns_to_forecast, filters,
                     train_test_split, padding_size)


if __name__ == "__main__":
    main()
