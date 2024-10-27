import json
import os.path

from keras.src.optimizers import Adam
from keras.src.saving.saving_api import load_model
from sklearn.metrics import mean_absolute_error

import eval_model
from data_generator import BaseDataset
from dataset import DatasetGenerator, FillnaTypes, XYType
from model import ModelTrainer, generate_model_name
from models_repo.LSTMRegressor import LSTMRegressor

if __name__ == '__main__':
    columns = ['date', 'RSAM', 'T_olb',
               'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
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
    train_test_split = 0.75
    loss = 'mae'
    save = True
    padding_size = 30
    dataset_generator = DatasetGenerator(columns=columns, seq_len_x=seq_len_x, seq_len_y=seq_len_y, data_path=data_path,
                                         encoders=encoders, scaler_path=scalers)
    df = dataset_generator.generate_frame()
    (X_train, y_train), (X_test, y_test) = dataset_generator.generate_XY(df=df,
                                                                         columns_to_scale=columns_to_scale,
                                                                         columns_to_drop=columns_to_drop,
                                                                         columns_to_forecast=columns_to_forecast,
                                                                         columns_to_filter=columns_to_filter,
                                                                         fill_na_type=FillnaTypes.MEAN,
                                                                         remove_not_known=False,
                                                                         type=XYType.TRAINTEST,
                                                                         train_test_split=train_test_split,
                                                                         padding_size=padding_size)
    # Divido Train
    (X_train, y_train), (X_valid, y_valid) = BaseDataset.split_train_valid((X_train, y_train), shuffle=False)
    model_name = generate_model_name()
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
    model = load_model(f'saved_model/{model_name}.x')
    # Adjust test set by excluding the overlap
    X_test_eval = X_test[padding_size:]
    y_test_eval = y_test[padding_size:]

    # Model predictions
    lstm_y_preds = regressor.model.predict(X_test_eval).reshape(-1)

    # Calculate Mean Absolute Error (MAE)
    mae = float(mean_absolute_error(y_test_eval.reshape(y_test_eval.shape[0]), lstm_y_preds))

    # Calculate Pearson's correlation => i have a filtered X
    pearsonsval = eval_model.eval(model_name, (X_test_eval, y_test_eval), target_column=-1)

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
                'filter_type': 'Low',
                'train_test_split': train_test_split,
                'padding_size': padding_size
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
