import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from collections import deque
import os
import joblib

class FactorAnalyzer:
    def __init__(self, df, all_keys):
        """
        Initialize the FactorAnalyzer with data and keys.
        
        Args:
            df (pd.DataFrame): Input dataframe containing the data
            all_keys (list): List of column names to use in the analysis
        """
        self.df = df
        self.all_keys = all_keys
        self.model_path = "/Users/kevynzhang/Downloads/system_output/factor_analysis_model.joblib"

        self.change_point = 300
        self.window_size = 50
        self.k = 10
        self.h = 200

    def perform_analysis(self):
        """
        Perform factor analysis and CUSUM monitoring on the data.
        
        Returns:
            tuple: (alert_curve, Composite_Score)
        """
        # Perform factor analysis
        all_data = self.df[self.all_keys]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_data)

        # PCA needs to be calculated every time
        pca = PCA()
        pca.fit(X_scaled)
        n_factors = sum(pca.explained_variance_ > 1)
        print(f"建议因子数量(Kaiser准则): {n_factors}")

        # Check if FA model exists
        if os.path.exists(self.model_path):
            print("Loading existing FactorAnalysis model...")
            fa = joblib.load(self.model_path)
        else:
            print("Training new FactorAnalysis model...")
            fa = FactorAnalysis(n_components=n_factors, rotation='varimax')
            fa.fit(X_scaled)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Save the model
            joblib.dump(fa, self.model_path)

        # Calculate factor scores and composite score
        factor_scores = fa.transform(X_scaled)
        weights = pca.explained_variance_ratio_[:n_factors]
        Composite_Score = np.dot(factor_scores, weights)
        self.df['fa'] = Composite_Score

        # Perform CUSUM monitoring
        self._perform_cusum_monitoring()

        return self.df['alert_curve'], Composite_Score

    def _perform_cusum_monitoring(self):
        """
        Perform CUSUM monitoring on the factor analysis scores.
        """
        change_point = self.change_point
        window_size = self.window_size
        mu_target = np.mean(self.df['fa'][:change_point])
        sigma0_sq = np.var(self.df['fa'][:change_point])
        k = self.k
        h = self.h

        window = deque(maxlen=window_size)
        S_window = 0
        cusum_vals = []
        alarms = []

        for t in range(len(self.df['fa'])):
            r_t = (self.df['fa'][t] - mu_target) ** 2

            if len(window) == window_size:
                S_window -= (window[0] / sigma0_sq - k)
            window.append(r_t)
            S_window += (r_t / sigma0_sq - k)

            S_window = max(0, S_window)
            cusum_vals.append(S_window)

            if S_window > h:
                alarms.append(t)

        self.all_keys.append('CUSUM')
        self.df['CUSUM'] = cusum_vals

        # Generate alert curve
        alert_curve = np.zeros(len(self.df['fa']))
        for alarm in alarms:
            alert_curve[alarm] = 1
        self.df['alert_curve'] = alert_curve 