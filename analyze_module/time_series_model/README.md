# Time Series Forecasting Model

Simplified time series forecasting pipeline with minimal logging and clean code structure.

## Files

- `model.py` - Core forecasting model classes (AR, ARIMA, VAR, SGD-AR)
- `forecasting.py` - Data loading, preprocessing, and pipeline orchestration
- `run_forecasting.py` - Simple script to run the complete pipeline
- `horizon_experiment.py` - Script to test different forecast horizons

## Usage

### Basic Forecasting
```bash
python run_forecasting.py
```

### Horizon Experiment
```bash
python horizon_experiment.py
```

### Command Line Interface
```bash
python forecasting.py --data_dir /path/to/data --model_dir /path/to/models --splits_file /path/to/splits.json --reduced_dir /path/to/reduced --model_type ar --forecast_horizon 10
```

## Pipeline Steps

1. **Reduce time series using FA** - Convert multi-dimensional data to 1D using Factor Analysis
2. **Train forecasting model** - Train AR/ARIMA/VAR/SGD-AR model dynamically on each file (incremental learning)
3. **Evaluate model** - Calculate MSE, MAE, RMSE between predicted and actual values dynamically on each test file

## Model Types

- `ar` - Autoregressive model
- `arima` - ARIMA model
- `var` - Vector Autoregressive model
- `sgd_ar` - Stochastic Gradient Descent AR model

## Output

- Trained models saved in `weights/` directory
- Reduced 1D data saved in `reduced_fa1/` directory
- Evaluation metrics printed to console
- Horizon experiment results saved as CSV
