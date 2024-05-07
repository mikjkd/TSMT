import errno
import hashlib
import os.path
import random

import pandas as pd
import yaml
from keras.src.optimizers import Adam

import eval_model
from data_generator import BaseDataset
from dataset import generate_dataset, FillnaTypes
# from dataset import generate_dataset
from model import ModelTrainer, LSTMRegressor


def generate_model_name(hyperparameters):
    # Convert hyperparameters to a string
    hyperparameters_str = str(hyperparameters) + str(random.random())
    # Generate SHA-256 hash
    hash_object = hashlib.sha256(hyperparameters_str.encode())
    model_name = hash_object.hexdigest()[:8]  # Take first 8 characters for readability

    return model_name


if __name__ == '__main__':
    with open('init.yaml', 'r') as file:
        init_data = yaml.safe_load(file)
    columns = ['date', 'RSAM', 'T_olb', 'Ru_olb', 'P_olb', 'Rn_olb', 'T_msa',
               'Ru_msa', 'P_msa', 'Rn_msa', 'displacement (cm)',
               'background seismicity']
    random.seed(200)
    # Iterate over configurations
    for config in init_data:
        hyperparameters = config['hyperparameters']
        missing_value_strategy = config['missing_value_strategy']
        remove_unknown_train_values = config['remove_unknown_train_values']
        remove_unknown_test_values = config['remove_unknown_test_values']
        window_in = hyperparameters['window_in']
        window_out = hyperparameters['window_out']
        batch_size = hyperparameters['batch_size']
        learning_rate = hyperparameters['learning_rate']
        loss = hyperparameters['loss']
        epochs = hyperparameters['epochs']

        model_name = generate_model_name(hyperparameters)

        # genero dataset in una cartella dedicata al modello.
        dataset_path = f'{os.getcwd()}/datasets/{model_name}/'
        try:
            os.mkdir(dataset_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory  già esistente.')
            else:
                exit()
        # genero training set
        generate_dataset(columns=columns, data_path='data/train.csv', base_path=f'datasets/{model_name}',
                         filename='train',
                         seq_len_x=window_in,
                         seq_len_y=window_out,
                         fill_na_type=FillnaTypes.from_string(missing_value_strategy),
                         remove_not_known=remove_unknown_train_values)

        # genero test set
        generate_dataset(columns=columns, data_path='data/test.csv', base_path=f'datasets/{model_name}',
                         filename='test',
                         seq_len_x=window_in,
                         seq_len_y=window_out,
                         fill_na_type=FillnaTypes.from_string(missing_value_strategy),
                         remove_not_known=remove_unknown_test_values)

        # X_train, X_test, y_train, y_test = load_data()
        data_path = f'datasets/{model_name}/dataset'

        # dataset
        dataset = BaseDataset(data_path=data_path)
        # trainer
        trainer = ModelTrainer(batch_size=batch_size)
        # carico i dati, li divido e creo i generators
        train_filenames, test_filenames = dataset.load_data(shuffle=False)
        # li carico già divisi, non serve più splittarli
        train_filenames, valid_filenames = dataset.split_train_valid(train_filenames, shuffle=True)
        train_generator, valid_generator, input_shape, output_shape = dataset.generate_data(train_filenames,
                                                                                            valid_filenames)

        # genero il modello a che prende in considerazione input ed output shape
        regressor = LSTMRegressor(model_name=model_name)
        regressor.generate_model(input_shape, output_shape)

        # alleno il modello
        trainer.run(
            model=regressor.model,
            model_name=regressor.model_name,
            train={"filenames": train_filenames, "generator": train_generator},
            test={'filenames': valid_filenames, 'generator': valid_generator},
            optimizer=Adam(learning_rate=learning_rate),
            epochs=epochs,
            loss=loss
        )

        print("accuracy test: ")
        _, test_generator, __, ___ = dataset.generate_data(train_filenames, test_filenames)
        lstm_y_preds = regressor.model.predict(test_generator)
        accuracy = regressor.model.evaluate(test_generator)

        print("pearsons: ")
        # eval_model.eval(model_name)
        X, y_true = dataset.generator_to_Xy(test_generator)
        corr, scaled_corr = eval_model.eval_pearsonsr(lstm_y_preds, y_true,
                                                      scalers_path=f'datasets/{model_name}/train-scalers')
        # Store configuration and results

        # scaler = joblib.load('')

        # plt.plot(scaled_y_true, label='true')
        # plt.plot(scaled_lstm_y_preds, label='preds')
        # plt.legend()
        # plt.show()

        config_result = {
            'model_name': model_name,
            'missing_value_strategy': missing_value_strategy,
            'remove_unknown_train_values': remove_unknown_train_values,
            'remove_unknown_test_values': remove_unknown_test_values,
            'accuracy': accuracy[0],
            'pearons_r': corr
        }
        for key, value in hyperparameters.items():
            config_result[key] = value
        # Append results to CSV file
        df = pd.DataFrame(config_result, index=[0])
        create_header = False if os.path.exists(os.getcwd() + '/results.csv') else True
        df.to_csv('results.csv', mode='a', index=False, header=create_header)
