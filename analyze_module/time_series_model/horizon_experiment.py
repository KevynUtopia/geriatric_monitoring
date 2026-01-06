#!/usr/bin/env python3
"""
Simple script to test different forecast horizons.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecasting import TimeSeriesForecastingPipeline
from model import TimeSeriesForecastingModel

def main():
    # Configuration
    data_dir = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/detection_output/results_alignment_soft/DATASET"
    model_dir = "/Users/kevynzhang/codespace/SNH/time_series_model/weights"
    splits_file = "/Users/kevynzhang/codespace/SNH/data_splits.json"
    reduced_dir = "/Users/kevynzhang/codespace/SNH/time_series_model/reduced_fa1"
    
    # Test different horizons
    horizons = [1, 3, 5, 7, 9, 10, 15, 20, 25, 30, 40]
    model_type = 'var'
    lag_order = 10
    
    print("=" * 60)
    print("FORECAST HORIZON EXPERIMENT")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TimeSeriesForecastingPipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        splits_file=splits_file,
        reduced_dir=reduced_dir
    )
    
    # Step 1: Reduce data
    pipeline.reduce_data()
    
    # Results storage
    results = []
    
    # Pick one example test series shared by all horizons
    pipeline.reduce_data()

    # Select a single test file to visualize across all horizons
    # Use the first available test file after reduction
    def pick_example_series():
        for person_id, filename, _df in pipeline.loader.iter_split_dfs('test'):
            reduced_path = os.path.join(reduced_dir, f"{os.path.splitext(filename)[0]}_fa1.csv")
            if os.path.exists(reduced_path):
                ex_df = pipeline.loader.load_reduced_file(reduced_path)
                if 'fa1' in ex_df.columns and len(ex_df) > 200:
                    return filename, ex_df
        return None, None

    example_filename, example_df = pick_example_series()
    if example_df is None:
        print("No suitable example series found for visualization.")
        return

    # Define a shared seed window and ground truth for comparison
    # Use last 200 points of the example as context; forecast horizon will vary
    context_len = 200
    seed_df = example_df.iloc[:context_len].reset_index(drop=True)
    full_truth = example_df.iloc[context_len:].reset_index(drop=True)

    # train and test each horizon
    for horizon in horizons:
        print(f"\nTesting horizon: {horizon}")
        
        # Train model fresh with specific horizon (dynamic over train files)
        pipeline.train_model(model_type=model_type, forecast_horizon=horizon, lag_order=lag_order)

        # Load model for direct use
        model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
        model = TimeSeriesForecastingModel(model_type=model_type, forecast_horizon=horizon, lag_order=lag_order)
        model.load_model(model_path)

        # Forecast from the shared seed window
        pred_df = model.forecast_from_seed(seed_df, steps=horizon)
        pred = pred_df['fa1'].values if 'fa1' in pred_df.columns else pred_df.iloc[:, 0].values
        truth = full_truth['fa1'].iloc[:horizon].values if 'fa1' in full_truth.columns else full_truth.iloc[:horizon, 0].values

        # Compute metrics for this horizon on the example series only
        mse = float(np.mean((truth - pred) ** 2))
        mae = float(np.mean(np.abs(truth - pred)))
        rmse = float(np.sqrt(mse))
        # Store example-only metrics under explicit names
        metrics = {
            'example_mse': mse,
            'example_mae': mae,
            'example_rmse': rmse,
        }

        # Also compute aggregate std across test files using the pipeline evaluator
        agg = pipeline.evaluate_model(model_type)
        if agg:
            for k in ['fa1_mse_mean', 'fa1_mse_std', 'fa1_mae_mean', 'fa1_mae_std', 'fa1_rmse_mean', 'fa1_rmse_std']:
                if k in agg:
                    metrics[k] = float(agg[k])
        
        # Step-wise error profile (per forecast step)
        step_errors = pd.DataFrame({
            'step': np.arange(1, horizon + 1),
            'abs_error': np.abs(truth - pred),
            'sq_error': (truth - pred) ** 2,
        })
        profile_dir = os.path.join(model_dir, 'horizon_step_profiles')
        os.makedirs(profile_dir, exist_ok=True)
        step_profile_path = os.path.join(profile_dir, f'step_profile_h{horizon}.csv')
        step_errors.to_csv(step_profile_path, index=False)
        
        # Store results
        if metrics:
            result = {
                'horizon': horizon,
                'model_type': model_type,
                'lag_order': lag_order
            }
            result.update(metrics)
            results.append(result)

        # Visualization for the example series (one figure per horizon)
        fig, ax = plt.subplots(figsize=(10, 4))
        t_context = np.arange(len(seed_df))
        t_forecast = np.arange(len(seed_df), len(seed_df) + horizon)
        
        ax.plot(t_context, seed_df['fa1'].values, label='Input (context)', color='#6baed6')
        ax.plot(t_forecast, truth, label='Ground Truth', color='#31a354')
        ax.plot(t_forecast, pred, label='Prediction', color='#de2d26')
        ax.set_title(f"Example Forecast | horizon={horizon} | file={example_filename}")
        ax.set_xlabel('Time (index)')
        ax.set_ylabel('fa1')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        viz_dir = os.path.join(model_dir, 'horizon_viz')
        os.makedirs(viz_dir, exist_ok=True)
        fig_path = os.path.join(viz_dir, f"example_h{horizon}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_file = os.path.join(model_dir, 'horizon_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to {results_file}")
        
        # Print summary
        print("\nSummary:")
        cols = [c for c in ['horizon', 'example_mse', 'example_mae', 'example_rmse', 'fa1_mse_mean', 'fa1_mse_std', 'fa1_mae_mean', 'fa1_mae_std', 'fa1_rmse_mean', 'fa1_rmse_std'] if c in results_df.columns]
        print(results_df[cols].to_string(index=False))
        print(f"\n✓ Visualizations saved to {os.path.join(model_dir, 'horizon_viz')}")
        print(f"✓ Step-wise error profiles saved to {os.path.join(model_dir, 'horizon_step_profiles')}")
    
    print("=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
