from dataset import generate_dataset
from eval_model import *
from model import model

if __name__ == '__main__':
    # Gemerate dataset
    generate_dataset()

    # Forecast Model
    lstm_model_name = 'lstm_model_no_zeros'
    regressor_model = model(lstm_model_name)

    # Eval model
    eval()
