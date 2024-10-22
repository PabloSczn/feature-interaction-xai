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

# Specify the dataset name
DATASET_NAME = 'friedman1'


def load_data_and_models():
    """
    Load the dataset and pre-trained models.

    Returns:
        tuple: A tuple containing the training features DataFrame and a dictionary of models.
    """
    # Load the dataset
    X_train = pd.read_csv('./data/friedman_X_train.csv')

    # Load the pre-trained models
    models = {}
    models['xgb'] = joblib.load('./models/xgb_model.pkl')
    models['rf'] = joblib.load('./models/rf_model.pkl')

    return X_train, models


def compute_H_statistic_pairwise(model, X, feature_j, feature_k, sample_size=500):
    """
    Compute the pairwise H-statistic for two features.

    Args:
        model: The trained model.
        X (pd.DataFrame): Feature dataset.
        feature_j (str): Name of the first feature.
        feature_k (str): Name of the second feature.
        sample_size (int, optional): Number of samples to use. Defaults to 500.

    Returns:
        float: The computed H2jk statistic.
    """
    # Sample data points to reduce computation
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]

    # Initialize arrays to store PD values
    PDjk = np.zeros(n)
    PDj = np.zeros(n)
    PDk = np.zeros(n)

    # Iterate over each sampled data point
    for i in tqdm(range(n), desc=f"Computing H-statistic for pair ({feature_j}, {feature_k})"):
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

    # Center the partial dependence functions
    PDjk_centered = PDjk - PDjk.mean()
    PDj_centered = PDj - PDj.mean()
    PDk_centered = PDk - PDk.mean()

    # Compute H2jk as per Friedman's definition
    numerator = np.sum((PDjk_centered - PDj_centered - PDk_centered) ** 2)
    denominator = np.sum(PDjk_centered ** 2)
    H2jk = numerator / denominator if denominator != 0 else np.nan

    return H2jk


def compute_H_statistic_total(model, X, feature_j, sample_size=500):
    """
    Compute the total H-statistic for a single feature interacting with all others.

    Args:
        model: The trained model.
        X (pd.DataFrame): Feature dataset.
        feature_j (str): Name of the feature.
        sample_size (int, optional): Number of samples to use. Defaults to 500.

    Returns:
        float: The computed H2j statistic.
    """
    # Sample data points to reduce computation
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]

    # Initialize arrays to store PD values
    f_x = model.predict(X_sample).reshape(-1)
    PDj = np.zeros(n)
    PD_minus_j = np.zeros(n)

    # Iterate over each sampled data point
    for i in tqdm(range(n), desc=f"Computing H-statistic for feature {feature_j}"):
        x_i = X_sample.iloc[i]

        # Partial dependence for feature_j
        X_j = X.copy()
        X_j[feature_j] = x_i[feature_j]
        preds_j = model.predict(X_j)
        PDj[i] = preds_j.mean()

        # Partial dependence excluding feature_j
        # Fix all features except feature_j at their values in x_i
        X_minus_j = pd.DataFrame(np.tile(x_i.values, (X.shape[0], 1)), columns=X.columns)
        X_minus_j[feature_j] = X[feature_j].values  # Vary feature_j
        preds_minus_j = model.predict(X_minus_j)
        PD_minus_j[i] = preds_minus_j.mean()

    # Center the functions
    f_x_centered = f_x - f_x.mean()
    PDj_centered = PDj - PDj.mean()
    PD_minus_j_centered = PD_minus_j - PD_minus_j.mean()

    # Compute H2j as per Friedman's definition
    numerator = np.sum((f_x_centered - PDj_centered - PD_minus_j_centered) ** 2)
    denominator = np.sum(f_x_centered ** 2)
    H2j = numerator / denominator if denominator != 0 else np.nan

    return H2j


def main():
    """
    Main function to compute and save H-statistics for all models.
    
    Explanations:
    - Pairwise H-statistics for all feature combinations.
    - Total H-statistics for each relevant feature.
    """
    # Load data and models
    X_train, models = load_data_and_models()

    # Identify relevant features based on the dataset
    if DATASET_NAME == 'friedman1':
        # Exclude noise features for Friedman 1 dataset
        relevant_features = X_train.columns[:5].tolist()
    else:
        relevant_features = X_train.columns.tolist()

    for model_name in MODEL_NAMES:
        print(f"\nProcessing model: {model_name.upper()}")
        model = models[model_name]

        # Initialize dictionaries to store H-statistics
        pairwise_H = {}
        total_H = {}

        # Compute pairwise H-statistics for all relevant feature combinations
        print("\nComputing pairwise H-statistics...")
        feature_pairs = list(combinations(relevant_features, 2))
        for (feature_j, feature_k) in tqdm(feature_pairs, desc="Pairwise H-statistics"):
            H2jk = compute_H_statistic_pairwise(model, X_train, feature_j, feature_k, sample_size=500)
            pairwise_H[(feature_j, feature_k)] = H2jk * 100  # Scale to percentage

        # Convert pairwise H to DataFrame and save
        pairwise_H_df = pd.DataFrame([
            {'Feature 1': pair[0], 'Feature 2': pair[1], 'H2jk (%)': f"{value:.2f}"}
            for pair, value in pairwise_H.items()
        ])
        pairwise_H_df.to_csv(os.path.join(EXPLANATIONS_DIR, model_name, 'pairwise_H_statistic.csv'), index=False)
        print(f"Pairwise H-statistics saved to {os.path.join(EXPLANATIONS_DIR, model_name, 'pairwise_H_statistic.csv')}")

        # Compute total H-statistics for each relevant feature
        print("\nComputing total H-statistics...")
        for feature_j in tqdm(relevant_features, desc="Total H-statistics"):
            H2j = compute_H_statistic_total(model, X_train, feature_j, sample_size=500)
            total_H[feature_j] = H2j * 100  # Scale to percentage

        # Convert total H to DataFrame and save
        total_H_df = pd.DataFrame([
            {'Feature': feature, 'H2j (%)': f"{value:.2f}"}
            for feature, value in total_H.items()
        ])
        total_H_df.to_csv(os.path.join(EXPLANATIONS_DIR, model_name, 'total_H_statistic.csv'), index=False)
        print(f"Total H-statistics saved to {os.path.join(EXPLANATIONS_DIR, model_name, 'total_H_statistic.csv')}")

        # Summary of results
        print("\nH-statistic computation completed.")
        print("Pairwise H-statistics indicate the strength of interaction between relevant feature pairs.")
        print("Total H-statistics indicate the overall interaction strength of each relevant feature with all others.")


if __name__ == "__main__":
    main()