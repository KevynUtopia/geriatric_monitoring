import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import SGDRegressor


class TimeSeriesForecastingModel:
    """Simplified time series forecasting model with minimal logging."""
    
    def __init__(self, model_type: str = 'ar', **kwargs):
        self.model_type = model_type
        self.forecast_horizon = kwargs.get('forecast_horizon', 10)
        self.lag_order = kwargs.get('lag_order', 10)
        self.is_fitted = False
        # ARIMA speed/conciseness controls
        self.arima_order = kwargs.get('arima_order', (1, 1, 1))
        self.arima_max_points = kwargs.get('arima_max_points', 2000)  # cap training window
        self.arima_maxiter = kwargs.get('arima_maxiter', 50)          # limit optimizer iterations
        
        # Model storage
        self.fitted_models = {}
        self.fitted_ar_models = {}
        self.sgd_models = {}
        self.last_values = {}
        self.feature_columns = []
    
    def fit(self, data: pd.DataFrame):
        """Fit the model to data (supports incremental learning)."""
        if self.model_type == 'ar':
            self._fit_ar(data)
        elif self.model_type == 'arima':
            self._fit_arima(data)
        elif self.model_type == 'var':
            self._fit_var(data)
        elif self.model_type == 'sgd_ar':
            self._fit_sgd_ar(data)
        self.is_fitted = True
    
    def _fit_ar(self, data: pd.DataFrame):
        """Fit AR model (supports incremental learning)."""
        ts_data = data.drop('timestamp', axis=1) if 'timestamp' in data.columns else data.copy()
        
        # Initialize feature columns on first fit
        if not self.feature_columns:
            self.feature_columns = ts_data.columns.tolist()
        
        for col in self.feature_columns:
            if col in ts_data.columns:
                series = ts_data[col].astype(float).dropna()
                if len(series) > self.lag_order:
                    # If model already exists, concatenate with existing data
                    if col in self.fitted_ar_models:
                        # Get existing data and concatenate
                        existing_data = self.fitted_ar_models[col].model.endog
                        combined_data = np.concatenate([existing_data, series.values])
                    else:
                        combined_data = series.values
                    
                    # Fit AR model on combined data
                    ar = AutoReg(combined_data, lags=self.lag_order, old_names=False)
                    self.fitted_ar_models[col] = ar.fit()
    
    def _fit_arima(self, data: pd.DataFrame):
        """Fit ARIMA model (supports incremental learning, fast/concise).

        Strategy:
        - Concatenate incoming data with existing endog (if any)
        - Keep only the last `self.arima_max_points` to cap cost
        - Fit ARIMA with small order and limited iterations
        """
        ts_data = data.drop('timestamp', axis=1) if 'timestamp' in data.columns else data.copy()
        
        # Initialize feature columns on first fit
        if not self.feature_columns:
            self.feature_columns = ts_data.columns.tolist()
        
        for col in self.feature_columns:
            if col in ts_data.columns:
                series = ts_data[col].astype(float).dropna()
                if len(series) > 10:
                    # If model already exists, concatenate with existing data
                    if col in self.fitted_models:
                        existing_data = self.fitted_models[col].data.orig_endog
                        combined_data = np.concatenate([existing_data, series.values])
                    else:
                        combined_data = series.values

                    # Keep only the most recent window to speed up fitting
                    if self.arima_max_points is not None and len(combined_data) > self.arima_max_points:
                        combined_data = combined_data[-self.arima_max_points:]

                    # Fit concise ARIMA
                    model = ARIMA(
                        combined_data,
                        order=tuple(self.arima_order),
                        trend='n',
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    # Limit iterations; suppress verbose output
                    self.fitted_models[col] = model.fit(method_kwargs={'maxiter': int(self.arima_maxiter), 'disp': 0})
    
    def _fit_var(self, data: pd.DataFrame):
        """Fit VAR model (supports incremental learning)."""
        ts_data = data.drop('timestamp', axis=1) if 'timestamp' in data.columns else data.copy()
        
        # Initialize feature columns on first fit
        if not self.feature_columns:
            self.feature_columns = ts_data.columns.tolist()
        
        if len(ts_data) > 20:
            # If model already exists, concatenate with existing data
            if 'var' in self.fitted_models:
                existing_data = self.fitted_models['var'].y
                combined_data = pd.concat([existing_data, ts_data], ignore_index=True)
            else:
                combined_data = ts_data
            
            # Fit VAR model on combined data
            model = VAR(combined_data)
            self.fitted_models['var'] = model.fit(maxlags=1)
    
    def _fit_sgd_ar(self, data: pd.DataFrame):
        """Fit SGD AR model."""
        ts_data = data.drop('timestamp', axis=1) if 'timestamp' in data.columns else data.copy()
        self.feature_columns = ts_data.columns.tolist()
        
        for col in self.feature_columns:
            series = ts_data[col].astype(float)
            X, y = self._generate_lagged_features(series, self.lag_order)
            if X.size > 0:
                self.sgd_models[col] = SGDRegressor(random_state=42)
                self.sgd_models[col].fit(X, y)
                self.last_values[col] = series.iloc[-self.lag_order:].values.tolist()
    
    def _generate_lagged_features(self, series: pd.Series, lag_order: int):
        """Generate lagged features for SGD AR."""
        values = series.values.astype(float)
        if len(values) <= lag_order:
            return np.empty((0, lag_order)), np.empty((0,))
        
        X_full = np.lib.stride_tricks.sliding_window_view(values, window_shape=lag_order)
        if X_full.shape[0] == 0:
            return np.empty((0, lag_order)), np.empty((0,))
        
        X = X_full[:-1]
        y = values[lag_order:]
        return X, y
    
    def forecast(self, steps: int) -> pd.DataFrame:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.model_type == 'ar':
            return self._forecast_ar(steps)
        elif self.model_type == 'arima':
            return self._forecast_arima(steps)
        elif self.model_type == 'var':
            return self._forecast_var(steps)
        elif self.model_type == 'sgd_ar':
            return self._forecast_sgd_ar(steps)
    
    def _forecast_ar(self, steps: int) -> pd.DataFrame:
        """Forecast using AR model."""
        forecasts = {}
        for col, res in self.fitted_ar_models.items():
            start = len(res.model.endog)
            end = start + steps - 1
            pred = res.predict(start=start, end=end, dynamic=False)
            forecasts[col] = np.asarray(pred)
        return pd.DataFrame(forecasts)

    def forecast_from_seed(self, seed_df: pd.DataFrame, steps: int) -> pd.DataFrame:
        """Forecast forward using a provided seed series instead of model endog.

        Supports AR and SGD-AR models. For AR, uses learned coefficients to roll
        forward from the provided seed values. Expects the seed_df to contain the
        same feature columns (e.g., 'fa1').
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        if 'timestamp' in seed_df.columns:
            ts_data = seed_df.drop('timestamp', axis=1)
        else:
            ts_data = seed_df.copy()

        forecasts: Dict[str, np.ndarray] = {}

        if self.model_type == 'sgd_ar':
            # Use the provided seed as the initial window and roll forward with learned SGD weights
            for col, model in self.sgd_models.items():
                if col not in ts_data.columns:
                    continue
                history = ts_data[col].astype(float).values.tolist()
                if len(history) < self.lag_order:
                    continue
                history = history[-self.lag_order:]
                pred_list: List[float] = []
                current_input = np.array(history, dtype=float).reshape(1, -1)
                for _ in range(steps):
                    next_pred = float(model.predict(current_input)[0])
                    pred_list.append(next_pred)
                    current_input = np.roll(current_input, -1)
                    current_input[0, -1] = next_pred
                forecasts[col] = np.asarray(pred_list)
            return pd.DataFrame(forecasts)

        if self.model_type == 'ar':
            for col, res in self.fitted_ar_models.items():
                if col not in ts_data.columns:
                    continue
                seed_values = ts_data[col].astype(float).values.tolist()
                if len(seed_values) < self.lag_order:
                    continue
                # Extract AR coefficients and constant
                params = res.params
                # Determine constant and AR lag coefficients based on trend
                trend = getattr(res.model, 'trend', 'c')
                k_ar = getattr(res.model, 'k_ar', self.lag_order)
                if trend == 'n' or len(params) == k_ar:
                    const = 0.0
                    ar_coefs = np.asarray(params[-k_ar:], dtype=float)
                else:
                    const = float(params[0])
                    ar_coefs = np.asarray(params[1:1 + k_ar], dtype=float)

                history = seed_values[-k_ar:]
                pred_list: List[float] = []
                for _ in range(steps):
                    next_val = const
                    # Sum phi_i * y_{t-i}
                    for i in range(k_ar):
                        next_val += ar_coefs[i] * history[-(i + 1)]
                    pred_list.append(next_val)
                    history = history[1:] + [next_val]
                forecasts[col] = np.asarray(pred_list)
            return pd.DataFrame(forecasts)

        # Fallback: use default forecasting if unsupported
        return self._forecast_ar(steps) if self.model_type == 'ar' else self.forecast(steps)
    
    def _forecast_arima(self, steps: int) -> pd.DataFrame:
        """Forecast using ARIMA model."""
        forecasts = {}
        for col, model in self.fitted_models.items():
            pred = model.forecast(steps=steps)
            forecasts[col] = np.asarray(pred)
        return pd.DataFrame(forecasts)
    
    def _forecast_var(self, steps: int) -> pd.DataFrame:
        """Forecast using VAR model."""
        pred = self.fitted_models['var'].forecast(self.fitted_models['var'].y, steps=steps)
        return pd.DataFrame(pred, columns=self.feature_columns)
    
    def _forecast_sgd_ar(self, steps: int) -> pd.DataFrame:
        """Forecast using SGD AR model."""
        forecasts = {}
        for col, model in self.sgd_models.items():
            last_vals = np.array(self.last_values[col]).reshape(1, -1)
            pred = []
            current_input = last_vals.copy()
            
            for _ in range(steps):
                next_pred = model.predict(current_input)[0]
                pred.append(next_pred)
                current_input = np.roll(current_input, -1)
                current_input[0, -1] = next_pred
            
            forecasts[col] = np.array(pred)
        return pd.DataFrame(forecasts)
    
    def evaluate(self, test_data: pd.DataFrame, forecast_horizon: int = None) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon
        
        # Prepare test data
        if 'timestamp' in test_data.columns:
            test_prepared = test_data.drop('timestamp', axis=1)
        else:
            test_prepared = test_data.copy()
        
        # Generate forecasts
        forecasts = self.forecast(forecast_horizon)
        
        # Calculate metrics
        metrics = {}
        for col in self.feature_columns:
            if col in forecasts.columns and col in test_prepared.columns:
                actual = test_prepared[col].dropna().iloc[:forecast_horizon]
                predicted = forecasts[col].iloc[:len(actual)]
                
                if len(actual) > 0 and len(predicted) > 0:
                    mse = mean_squared_error(actual, predicted)
                    mae = mean_absolute_error(actual, predicted)
                    rmse = np.sqrt(mse)
                    
                    metrics[f'{col}_mse'] = mse
                    metrics[f'{col}_mae'] = mae
                    metrics[f'{col}_rmse'] = rmse
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon,
            'lag_order': self.lag_order,
            'fitted_models': self.fitted_models,
            'fitted_ar_models': self.fitted_ar_models,
            'sgd_models': self.sgd_models,
            'last_values': self.last_values,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model_type = model_data['model_type']
        self.forecast_horizon = model_data.get('forecast_horizon', self.forecast_horizon)
        self.lag_order = model_data.get('lag_order', self.lag_order)
        self.fitted_models = model_data.get('fitted_models', {})
        self.fitted_ar_models = model_data.get('fitted_ar_models', {})
        self.sgd_models = model_data.get('sgd_models', {})
        self.last_values = model_data.get('last_values', {})
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']