from data_generator import BaseDataset
from dataset import generate_dataset
from model import ModelTrainer, LSTMRegressor

if __name__ == '__main__':
    # genero il dataset
    generate_dataset()
    # dataset
    dataset = BaseDataset(data_path='dataset/filenames.npy')
    # trainer
    trainer = ModelTrainer(batch_size=64)

    # carico i dati, li divido e creo i generators
    data = dataset.load_data(shuffle=False)
    train_filenames, test_filenames = dataset.split_data(data)
    train_generator, test_generator, input_shape, output_shape = dataset.generate_data(train_filenames, test_filenames)

    # genero il modello a che prende in considerazione input ed output shape
    lstm_model_name = 'lstm_model_no_zeros'
    lstm_regressor = LSTMRegressor(model_name=lstm_model_name)
    lstm_regressor.generate_model(input_shape, output_shape)

    # alleno il modello
    trainer.run(
        model=lstm_regressor.model,
        model_name=lstm_regressor.model_name,
        train={"filenames": train_filenames, "generator": train_generator},
        test={'filenames': test_filenames, 'generator': test_generator},
        shapes={'inout': input_shape, 'output': output_shape}
    )

    lstm_y_preds = lstm_regressor.model.predict(test_generator)
    lstm_regressor.model.evaluate(test_generator)
