# TimeSeriesModelingToolkit (TSMT)

A versatile toolkit for analyzing, preprocessing, and modeling time-series data. This framework provides tools for data scaling, filtering, sequence generation, and training various LSTM models, including both standard and `TimeDistributed` approaches.

## Key Features

- **Comprehensive Data Processing**: Scale, filter, and prepare time-series data for supervised learning.
- **Flexible Model Architectures**: Choose between standard and `TimeDistributed` LSTM models, with advanced configurations including attention mechanisms and residual connections.
- **Batch Data Generation**: Efficiently load data in batches for large datasets to optimize memory usage.
- **Model Training and Evaluation**: Use built-in training configurations with callbacks for early stopping, learning rate reduction, and model checkpointing.

## Repository Structure

- **`dataset.py`**: Contains the `Dataset` class for data preprocessing, scaling, and sequence generation for model training.
- **`data_generator.py`**: Manages data batching, enabling efficient loading for large time-series datasets.
- **`LSTMRegressor.py`**: Implements various LSTM models, including both standard and `TimeDistributed` architectures.
- **`main.py`**: Configures and initiates model training and evaluation, allowing for easy experimentation with different models and settings.
- **`model.py`**: Defines the `ModelTrainer` class, which supports early stopping, model checkpointing, and learning rate adjustments.
- **`eval_model.py`**: Provides evaluation functions to assess model performance, including metrics such as MAE and Pearson’s correlation.

## Setup

### Requirements

Install the required packages by running:

```bash
pip install -r requirements.txt
```

## Folder Structure
To organize saved models, scalers, and data files, follow this folder structure:

```text
TimeSeriesModelingToolkit/
├── dataset.py
├── data_generator.py
├── LSTMRegressor.py
├── main.py
├── model.py
├── eval_model.py
├── saved_model/
├── scalers/
└── data/
```

## Usage


1. **Data Preparation**: Place your raw data in the data/ folder. Configure sequence lengths and other parameters in main.py.

2. **Training the Model**: Run the following command to start training:
```bash
python main.py
```

3. **Evaluation**: After training, `main.py` will save evaluation metrics and generate visualizations. The `eval_model.py` script allows customization of evaluation metrics.

4. **Model Customization**: `LSTMRegressor.py` provides different LSTM configurations, including both standard and TimeDistributed options with advanced features like attention.

## Example Visualization

The toolkit includes tools for generating visualizations, comparing true vs. predicted values across sequences to evaluate model performance.

## Future Improvements ##

- Experiment with different attention mechanisms for improved accuracy.
- Extend model options with additional sequence models (e.g., GRU, Transformer).
- Explore hyperparameter tuning for dataset-specific performance optimization.