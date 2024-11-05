from keras.src.optimizers import Adam

import eval_model
from data_generator import BaseDataset
# from dataset import generate_dataset
from models_repo.model import ModelTrainer, generate_model_name
from models_repo.LSTMRegressor import LSTMRegressor

if __name__ == '__main__':
    # genero il dataset
    # generate_dataset()

    ### configuration
    data_path = 'dataset'
    batch_size = 64
    learning_rate = 0.001
    loss = 'mae'
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
    model_name = generate_model_name()
    regressor = LSTMRegressor(model_name=model_name)
    regressor.generate_model(input_shape, output_shape)

    # alleno il modello
    trainer.run(
        model=regressor.model,
        model_name=regressor.model_name,
        train={"filenames": train_filenames, "generator": train_generator},
        valid={'filenames': valid_filenames, 'generator': valid_generator},
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )

    _, test_generator, __, ___ = dataset.generate_data(train_filenames, test_filenames)
    lstm_y_preds = regressor.model.predict(test_generator)
    regressor.model.evaluate(test_generator)

    eval_model.eval(model_name, test_generator)
