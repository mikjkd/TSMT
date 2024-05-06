import hashlib
import os.path

from keras.src.optimizers import Adam

import eval_model
from data_generator import BaseDataset
# from dataset import generate_dataset
from model import ModelTrainer, LSTMRegressor
from dataset import generate_dataset, FillnaTypes
import pandas as pd
import yaml
import random

random.seed(10)


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
    # Iterate over configurations
    for config in init_data:
        hyperparameters = config['hyperparameters']
        missing_value_strategy = config['missing_value_strategy']
        remove_unknown_values = config['remove_unknown_values']
        window_in = hyperparameters['window_in']
        window_out = hyperparameters['window_out']
        generate_dataset(seq_len_x=window_in, seq_len_y=window_out,
                         fill_na_type=FillnaTypes.from_string(missing_value_strategy),
                         remove_not_known=remove_unknown_values
                         )

        # X_train, X_test, y_train, y_test = load_data()
        data_path = 'dataset'
        batch_size = hyperparameters['batch_size']
        learning_rate = hyperparameters['learning_rate']
        loss = hyperparameters['loss']
        epochs = hyperparameters['epochs']
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
        model_name = generate_model_name(hyperparameters)
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


        _, test_generator, __, ___ = dataset.generate_data(train_filenames, test_filenames)
        lstm_y_preds = regressor.model.predict(test_generator)
        accuracy = regressor.model.evaluate(test_generator)

        # eval_model.eval(model_name)
        X, y_true = dataset.generator_to_Xy(test_generator)
        corr, scaled_corr = eval_model.eval_pearsonsr(lstm_y_preds, y_true)
        # Store configuration and results
        config_result = {
            'model_name': model_name,
            'missing_value_strategy': missing_value_strategy,
            'remove_unknown_values': remove_unknown_values,
            'accuracy': accuracy[0],
            'pearons_r': corr
        }
        for key, value in hyperparameters.items():
            config_result[key] = value
        # Append results to CSV file
        df = pd.DataFrame(config_result, index=[0])
        create_header = False if os.path.exists(os.getcwd() + '/results.csv') else True
        df.to_csv('results.csv', mode='a', index=False, header=create_header)
