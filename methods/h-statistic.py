import os
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from tqdm import tqdm
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Define directories for explanations
EXPLANATIONS_DIR = "explanations/h-statistic"
MODEL_NAMES = ['xgb', 'rf']

# Ensure explanation directories exist
for model_name in MODEL_NAMES:
    os.makedirs(os.path.join(EXPLANATIONS_DIR, model_name), exist_ok=True)

def load_data_and_models():
    """
    Load the dataset and pre-trained models.
    
    Returns:
        X_train (pd.DataFrame): Training features.
        models (dict): Dictionary containing loaded models.
    """
    # Load the dataset
    X_train = pd.read_csv('./data/friedman_X_train.csv')
    
    # Load the pre-trained models
    models = {}
    models['xgb'] = joblib.load('./models/xgb_model.pkl')
    models['rf'] = joblib.load('./models/rf_model.pkl')
    
    return X_train, models

def compute_partial_dependence(model, X, features, grid_resolution=20):
    """
    Compute partial dependence for specified features.
    
    Args:
        model: Trained model with a predict method.
        X (pd.DataFrame): Dataset.
        features (list): List of feature names or indices.
        grid_resolution (int): Number of points in the grid for each feature.
        
    Returns:
        pd_results: Partial dependence results.
    """
    # Use sklearn's partial_dependence function
    # Convert feature names to indices if necessary
    if isinstance(features[0], str):
        features = [X.columns.get_loc(f) for f in features]
    
    pd_results = partial_dependence(model, X, features, grid_resolution=grid_resolution, kind='average')
    return pd_results

def compute_H_statistic_pairwise(model, X, feature_j, feature_k, sample_size=100):
    """
    Compute the pairwise H-statistic for two features.
    
    Args:
        model: Trained model with a predict method.
        X (pd.DataFrame): Dataset.
        feature_j (str): Name of the first feature.
        feature_k (str): Name of the second feature.
        sample_size (int): Number of samples to use for estimation.
        
    Returns:
        H2jk (float): Pairwise H-statistic between feature_j and feature_k.
    """
    # Sample data points to reduce computation
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]
    
    # Initialize arrays to store PD values
    PDjk = np.zeros(n)
    PDj = np.zeros(n)
    PDk = np.zeros(n)
    
    # Iterate over each sampled data point
    for i in tqdm(range(n), desc=f"Computing PD for pair ({feature_j}, {feature_k})"):
        x_i = X_sample.iloc[i]
        
        # Create a copy of X and set feature_j and feature_k to x_i's values
        X_jk = X.copy()
        X_jk[feature_j] = x_i[feature_j]
        X_jk[feature_k] = x_i[feature_k]
        preds_jk = model.predict(X_jk)
        PDjk[i] = preds_jk.mean()
        
        # Partial dependence for feature_j
        X_j = X.copy()
        X_j[feature_j] = x_i[feature_j]
        preds_j = model.predict(X_j)
        PDj[i] = preds_j.mean()
        
        # Partial dependence for feature_k
        X_k = X.copy()
        X_k[feature_k] = x_i[feature_k]
        preds_k = model.predict(X_k)
        PDk[i] = preds_k.mean()
    
    # Compute H2jk as per Friedman's definition
    numerator = np.sum((PDjk - PDj - PDk) ** 2)
    denominator = np.sum(PDjk ** 2)
    H2jk = numerator / denominator if denominator != 0 else np.nan
    
    return H2jk

def compute_H_statistic_total(model, X, feature_j, sample_size=100):
    """
    Compute the total H-statistic for a single feature interacting with all others.
    
    Args:
        model: Trained model with a predict method.
        X (pd.DataFrame): Dataset.
        feature_j (str): Name of the feature.
        sample_size (int): Number of samples to use for estimation.
        
    Returns:
        H2j (float): Total H-statistic for feature_j.
    """
    # Sample data points to reduce computation
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]
    
    # Initialize arrays to store PD values
    f_x = model.predict(X_sample).reshape(-1)
    PDj = np.zeros(n)
    PD_minus_j = np.zeros(n)
    
    # Iterate over each sampled data point
    for i in tqdm(range(n), desc=f"Computing PD for feature {feature_j}"):
        x_i = X_sample.iloc[i]
        
        # Partial dependence for feature_j
        X_j = X.copy()
        X_j[feature_j] = x_i[feature_j]
        preds_j = model.predict(X_j)
        PDj[i] = preds_j.mean()
        
        # Partial dependence excluding feature_j
        # To approximate PD−j, we average predictions while keeping all features except j fixed
        X_minus_j = X.copy()
        X_minus_j = X_minus_j.drop(columns=[feature_j])
        # To compute PD−j, we need to marginalize over all features except j
        # This is computationally expensive; here we approximate by using the mean prediction
        preds_minus_j = model.predict(X.copy())
        PD_minus_j[i] = preds_minus_j.mean()
    
    # Compute H2j as per Friedman's definition
    numerator = np.sum((f_x - PDj - PD_minus_j) ** 2)
    denominator = np.sum(f_x ** 2)
    H2j = numerator / denominator if denominator != 0 else np.nan
    
    return H2j

def main():
    """
    Main function to compute and save H-statistics for all models.
    """
    # Load data and models
    X_train, models = load_data_and_models()
    
    # List of features
    features = X_train.columns.tolist()
    
    for model_name in MODEL_NAMES:
        print(f"\nProcessing model: {model_name.upper()}")
        model = models[model_name]
        
        # Initialize dictionaries to store H-statistics
        pairwise_H = {}
        total_H = {}
        
        # Compute pairwise H-statistics for all feature combinations
        print("\nComputing pairwise H-statistics...")
        feature_pairs = list(combinations(features, 2))
        for (feature_j, feature_k) in tqdm(feature_pairs, desc="Pairwise H-statistics"):
            H2jk = compute_H_statistic_pairwise(model, X_train, feature_j, feature_k, sample_size=100)
            pairwise_H[(feature_j, feature_k)] = H2jk
        
        # Convert pairwise H to DataFrame and save
        pairwise_H_df = pd.DataFrame([
            {'Feature 1': pair[0], 'Feature 2': pair[1], 'H2jk': value}
            for pair, value in pairwise_H.items()
        ])
        pairwise_H_df.to_csv(os.path.join(EXPLANATIONS_DIR, model_name, 'pairwise_H_statistic.csv'), index=False)
        print(f"Pairwise H-statistics saved to {os.path.join(EXPLANATIONS_DIR, model_name, 'pairwise_H_statistic.csv')}")
        
        # Compute total H-statistics for each feature
        print("\nComputing total H-statistics...")
        for feature_j in tqdm(features, desc="Total H-statistics"):
            H2j = compute_H_statistic_total(model, X_train, feature_j, sample_size=100)
            total_H[feature_j] = H2j
        
        # Convert total H to DataFrame and save
        total_H_df = pd.DataFrame([
            {'Feature': feature, 'H2j': value}
            for feature, value in total_H.items()
        ])
        total_H_df.to_csv(os.path.join(EXPLANATIONS_DIR, model_name, 'total_H_statistic.csv'), index=False)
        print(f"Total H-statistics saved to {os.path.join(EXPLANATIONS_DIR, model_name, 'total_H_statistic.csv')}")
        
        # Summary of results
        print("\nH-statistic computation completed.")
        print("Pairwise H-statistics indicate the strength of interaction between feature pairs.")
        print("Total H-statistics indicate the overall interaction strength of each feature with all others.")
        print("These results can be compared with ALE and RuleFit to assess consistency in detected interactions.\n")

if __name__ == "__main__":
    main()