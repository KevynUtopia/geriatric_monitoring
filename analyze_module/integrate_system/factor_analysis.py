import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from collections import deque
import os
import joblib

class FactorAnalyzer:
    def __init__(self, df, all_keys, person_id=None, model_base_path=""):
        """
        Initialize the FactorAnalyzer with data and keys.
        
        Args:
            df (pd.DataFrame): Input dataframe containing the data
            all_keys (list): List of column names to use in the analysis
            person_id (str): Person identifier (e.g., 'p_1', 'p_2')
            model_base_path (str): Base path where models are stored
        """
        self.df = df
        self.all_keys = all_keys
        self.person_id = person_id
        self.model_base_path = model_base_path
        
        # Set model path based on person_id
        if person_id and model_base_path:
            self.model_path = os.path.join(model_base_path, f"fa_model_{person_id}.joblib")
        else:
            self.model_path = ""

        self.change_point = 300
        self.window_size = 50
        self.k = 10
        self.h = 200
        
    def _extract_person_id_from_data(self):
        """
        Try to extract person ID from the dataframe or other context.
        This is a fallback method if person_id is not provided.
        
        Returns:
            str: Extracted person ID or None
        """
        # This could be enhanced based on your data structure
        # For now, it's a placeholder
        return None

    def perform_analysis(self, var_window_size = 100, var_threshold = 2.):
        """
        Perform factor analysis and CUSUM monitoring on the data.
        
        Returns:
            tuple: (alert_curve, Composite_Score)
        """
        # Perform factor analysis
        all_data = self.df[self.all_keys]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_data)

        # PCA needs to be calculated every time for optimal factor number
        pca = PCA()
        pca.fit(X_scaled)
        n_factors = sum(pca.explained_variance_ > 1)
        print(f"建议因子数量(Kaiser准则): {n_factors}")

        # Check if person-specific FA model exists
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading existing FactorAnalysis model for {self.person_id}: {self.model_path}")
            try:
                model_data = joblib.load(self.model_path)
                
                # Extract components from saved model data
                if isinstance(model_data, dict):
                    fa = model_data['fa_model']
                    saved_scaler = model_data.get('scaler')
                    saved_columns = model_data.get('activity_columns', self.all_keys)
                    
                    # Verify column compatibility (only check number of columns, not names)
                    if len(saved_columns) != len(self.all_keys):
                        print(f"Warning: Saved model has {len(saved_columns)} columns but current data has {len(self.all_keys)} columns")
                        print("Training new model instead...")
                        fa = self._train_new_model(X_scaled, n_factors)
                    else:
                        # Use saved scaler if available and column count matches
                        if saved_scaler is not None:
                            X_scaled = saved_scaler.transform(all_data)
                        print(f"Successfully loaded model with {fa.n_components} factors (columns: {len(saved_columns)} -> {len(self.all_keys)})")
                else:
                    # Legacy format - just the FA model
                    fa = model_data
                    print(f"Loaded legacy model format with {fa.n_components} factors")
                    
            except Exception as e:
                print(f"Error loading model {self.model_path}: {e}")
                print("Training new model instead...")
                fa = self._train_new_model(X_scaled, n_factors)
        else:
            if self.person_id:
                print(f"No existing model found for {self.person_id}, training new model...")
            else:
                print("Training new FactorAnalysis model...")
            fa = self._train_new_model(X_scaled, n_factors)

        # Calculate factor scores and composite score
        factor_scores = fa.transform(X_scaled)
        
        # Use the actual number of factors from the model
        actual_n_factors = fa.n_components
        weights = pca.explained_variance_ratio_[:actual_n_factors]
        
        # Ensure weights match the number of factor scores
        if len(weights) > factor_scores.shape[1]:
            weights = weights[:factor_scores.shape[1]]
        elif len(weights) < factor_scores.shape[1]:
            # Pad with smaller weights if needed
            additional_weights = pca.explained_variance_ratio_[len(weights):factor_scores.shape[1]]
            weights = np.concatenate([weights, additional_weights])
        
        Composite_Score = np.dot(factor_scores, weights)

        # Perform CUSUM monitoring
        alert_curve = self._perform_cusum_monitoring(Composite_Score)

        # give another curve of Composite_Score about the variance of the data through a window of 100
        variance_curve = np.zeros(len(Composite_Score))
        for i in range(len(Composite_Score) - var_window_size + 1):
            # Calculate variance for window [i:i+var_window_size]
            window_variance = np.var(Composite_Score[i:i+var_window_size])
            # Assign to middle timestamp of the window
            middle_idx = i + var_window_size // 2
            variance_curve[middle_idx] = window_variance
        # retain variance only larger than var_threshold, otherwise set to 0
        variance_curve = np.where(variance_curve > var_threshold, variance_curve, 0)
        return alert_curve, Composite_Score, variance_curve
    
    def _train_new_model(self, X_scaled, n_factors):
        """
        Train a new Factor Analysis model.
        
        Args:
            X_scaled (np.ndarray): Scaled input data
            n_factors (int): Number of factors to use
            
        Returns:
            FactorAnalysis: Trained model
        """
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(X_scaled)
        
        # Save the model if path is specified
        if self.model_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                # Save model with metadata
                scaler = StandardScaler()
                scaler.fit(self.df[self.all_keys])
                
                model_data = {
                    'fa_model': fa,
                    'scaler': scaler,
                    'n_factors': n_factors,
                    'activity_columns': self.all_keys,
                    'training_samples': len(self.df),
                    'person_id': self.person_id
                }
                
                joblib.dump(model_data, self.model_path)
                print(f"Saved new model to: {self.model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        return fa

    def _perform_cusum_monitoring(self, Composite_Score):
        """
        Perform CUSUM monitoring on the factor analysis scores.
        """
        change_point = self.change_point
        window_size = self.window_size
        mu_target = np.mean(Composite_Score[:change_point])
        sigma0_sq = np.var(Composite_Score[:change_point])
        k = self.k
        h = self.h

        window = deque(maxlen=window_size)
        S_window = 0
        cusum_vals = []
        alarms = []

        for t in range(len(Composite_Score)):
            r_t = (Composite_Score[t] - mu_target) ** 2

            if len(window) == window_size:
                S_window -= (window[0] / sigma0_sq - k)
            window.append(r_t)
            S_window += (r_t / sigma0_sq - k)

            S_window = max(0, S_window)
            cusum_vals.append(S_window)

            if S_window > h:
                alarms.append(t)

        # Generate alert curve
        alert_curve = np.zeros(len(Composite_Score))
        for alarm in alarms:
            alert_curve[alarm] = 1
        return alert_curve