#!/usr/bin/env python3
"""
Simple script to run time series forecasting pipeline.
"""

import os
import sys
from forecasting import TimeSeriesForecastingPipeline

def main():
    # Configuration
    data_dir = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/detection_output/results_alignment_soft/DATASET"
    model_dir = "/Users/kevynzhang/codespace/SNH/time_series_model/weights"
    splits_file = "/Users/kevynzhang/codespace/SNH/data_splits.json"
    reduced_dir = "/Users/kevynzhang/codespace/SNH/time_series_model/reduced_fa1"
    
    # Model parameters
    model_type = 'ar'  # Options: 'ar', 'arima', 'var', 'sgd_ar'
    forecast_horizon = 10
    lag_order = 10
    
    print("=" * 60)
    print("TIME SERIES FORECASTING PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TimeSeriesForecastingPipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        splits_file=splits_file,
        reduced_dir=reduced_dir
    )
    
    # Run pipeline
    pipeline.reduce_data()
    pipeline.train_model(
        model_type=model_type,
        forecast_horizon=forecast_horizon,
        lag_order=lag_order
    )
    pipeline.evaluate_model(model_type)
    
    print("=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
