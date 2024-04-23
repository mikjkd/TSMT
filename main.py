from data_generator import BaseDataset
from dataset import generate_dataset
from model import ModelTrainer, LSTMRegressor, LinearRegressor

if __name__ == '__main__':
    # genero il dataset
    generate_dataset()
    # dataset
    dataset = BaseDataset(data_path='dataset/filenames.npy')
    # trainer
    trainer = ModelTrainer(batch_size=64)

    # carico i dati, li divido e creo i generators
    train_filenames, test_filenames = dataset.load_data(shuffle=False)
    train_generator, test_generator, input_shape, output_shape = dataset.generate_data(train_filenames, test_filenames)

    # genero il modello a che prende in considerazione input ed output shape
    model_name = 'lin_model_no_zeros'
    regressor = LinearRegressor(model_name=model_name)
    regressor.generate_model(input_shape, output_shape)

    # alleno il modello
    trainer.run(
        model=regressor.model,
        model_name=regressor.model_name,
        train={"filenames": train_filenames, "generator": train_generator},
        test={'filenames': test_filenames, 'generator': test_generator},
    )

    lstm_y_preds = regressor.model.predict(test_generator)
    regressor.model.evaluate(test_generator)
