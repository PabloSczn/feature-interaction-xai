import os
import sys
import logging
from itertools import combinations

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_NAME = 'friedman1'
DATA_PATH = './data/friedman_X_train.csv'
MODEL_PATHS = {
    'xgb': './models/xgb_model.pkl',
    'rf': './models/rf_model.pkl'
}
EXPLANATIONS_DIR = "explanations/h-statistic"
MODEL_NAMES = ['xgb', 'rf']
SAMPLE_SIZE = 500

def create_directories(base_dir, model_names):
    """
    Create directories for saving H-statistic explanations for each model.

    Args:
        base_dir (str): Base directory path.
        model_names (list): List of model names.

    Returns:
        dict: Mapping of model names to their respective directories.
    """
    directories = {}
    try:
        for model_name in model_names:
            model_dir = os.path.join(base_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            directories[model_name] = model_dir
            logger.debug(f"Directory ensured: {model_dir}")
        return directories
    except Exception as e:
        logger.error(f"Failed to create directories in {base_dir}: {e}")
        raise

def load_data(data_path):
    """
    Load the dataset from the specified CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded feature dataset.
    """
    try:
        logger.info(f"Loading dataset from {data_path}...")
        X_train = pd.read_csv(data_path)
        logger.info(f"Dataset loaded with shape {X_train.shape}.")
        return X_train
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in the file {data_path}.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

def load_models(model_paths):
    """
    Load pre-trained models from the specified paths.

    Args:
        model_paths (dict): Mapping of model names to their file paths.

    Returns:
        dict: Loaded model objects.
    """
    models = {}
    try:
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}.")
            logger.info(f"Loading model '{name}' from {path}...")
            model = joblib.load(path)
            models[name] = model
            logger.info(f"Model '{name}' loaded successfully.")
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def compute_h_statistic_pairwise(model, X, feature_j, feature_k, sample_size=SAMPLE_SIZE):
    """
    Compute the pairwise H-statistic for two features.

    Args:
        model: Trained model.
        X (pd.DataFrame): Feature dataset.
        feature_j (str): Name of the first feature.
        feature_k (str): Name of the second feature.
        sample_size (int): Number of samples to use.

    Returns:
        float: Pairwise H-statistic scaled to percentage.
    """
    logger.debug(f"Computing pairwise H-statistic for features: {feature_j}, {feature_k}")
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]

    PDjk = np.zeros(n)
    PDj = np.zeros(n)
    PDk = np.zeros(n)

    for i in tqdm(range(n), desc=f"Pair ({feature_j}, {feature_k})"):
        x_i = X_sample.iloc[i]

        # Compute PDjk
        X_jk = X.copy()
        X_jk[feature_j] = x_i[feature_j]
        X_jk[feature_k] = x_i[feature_k]
        preds_jk = model.predict(X_jk)
        PDjk[i] = preds_jk.mean()

        # Compute PDj
        X_j = X.copy()
        X_j[feature_j] = x_i[feature_j]
        preds_j = model.predict(X_j)
        PDj[i] = preds_j.mean()

        # Compute PDk
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
    H2jk = (numerator / denominator) * 100 if denominator != 0 else np.nan

    logger.debug(f"Pairwise H-statistic for ({feature_j}, {feature_k}): {H2jk:.2f}%")
    return H2jk

def compute_h_statistic_total(model, X, feature_j, sample_size=SAMPLE_SIZE):
    """
    Compute the total H-statistic for a single feature interacting with all others.

    Args:
        model: Trained model.
        X (pd.DataFrame): Feature dataset.
        feature_j (str): Feature name.
        sample_size (int): Number of samples to use.

    Returns:
        float: Total H-statistic scaled to percentage.
    """
    logger.debug(f"Computing total H-statistic for feature: {feature_j}")
    X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    n = X_sample.shape[0]

    f_x = model.predict(X_sample).reshape(-1)
    PDj = np.zeros(n)
    PD_minus_j = np.zeros(n)

    for i in tqdm(range(n), desc=f"Feature {feature_j}"):
        x_i = X_sample.iloc[i]

        # Compute PDj
        X_j = X.copy()
        X_j[feature_j] = x_i[feature_j]
        preds_j = model.predict(X_j)
        PDj[i] = preds_j.mean()

        # Compute PD_minus_j
        X_minus_j = X.copy()
        for col in X.columns:
            if col != feature_j:
                X_minus_j[col] = x_i[col]
        preds_minus_j = model.predict(X_minus_j)
        PD_minus_j[i] = preds_minus_j.mean()

    # Center the functions
    f_x_centered = f_x - f_x.mean()
    PDj_centered = PDj - PDj.mean()
    PD_minus_j_centered = PD_minus_j - PD_minus_j.mean()

    # Compute H2j as per Friedman's definition
    numerator = np.sum((f_x_centered - PDj_centered - PD_minus_j_centered) ** 2)
    denominator = np.sum(f_x_centered ** 2)
    H2j = (numerator / denominator) * 100 if denominator != 0 else np.nan

    logger.debug(f"Total H-statistic for {feature_j}: {H2j:.2f}%")
    return H2j

def save_h_statistics(pairwise_h, total_h, save_dir):
    """
    Save the computed H-statistics to CSV files.

    Args:
        pairwise_h (dict): Pairwise H-statistics.
        total_h (dict): Total H-statistics.
        save_dir (str): Directory to save the CSV files.
    """
    try:
        # Save Pairwise H-statistics
        pairwise_h_df = pd.DataFrame([
            {'Feature 1': pair[0], 'Feature 2': pair[1], 'H2jk (%)': f"{value:.2f}"}
            for pair, value in pairwise_h.items()
        ])
        pairwise_path = os.path.join(save_dir, 'pairwise_H_statistic.csv')
        pairwise_h_df.to_csv(pairwise_path, index=False)
        logger.info(f"Pairwise H-statistics saved to {pairwise_path}")

        # Save Total H-statistics
        total_h_df = pd.DataFrame([
            {'Feature': feature, 'H2j (%)': f"{value:.2f}"}
            for feature, value in total_h.items()
        ])
        total_path = os.path.join(save_dir, 'total_H_statistic.csv')
        total_h_df.to_csv(total_path, index=False)
        logger.info(f"Total H-statistics saved to {total_path}")
    except Exception as e:
        logger.error(f"Failed to save H-statistics: {e}")
        raise

def compute_and_save_h_statistics(model, model_name, X_train, save_dir):
    """
    Compute and save both pairwise and total H-statistics for a given model.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Feature dataset.
        save_dir (str): Directory to save the H-statistics.
    """
    logger.info(f"Processing model: {model_name.upper()}")

    # Determine relevant features
    if DATASET_NAME == 'friedman1':
        relevant_features = X_train.columns[:5].tolist()
    else:
        relevant_features = X_train.columns.tolist()

    pairwise_H = {}
    total_H = {}

    # Compute Pairwise H-statistics
    logger.info("Computing pairwise H-statistics...")
    feature_pairs = list(combinations(relevant_features, 2))
    for feature_j, feature_k in tqdm(feature_pairs, desc="Pairwise H-statistics"):
        H2jk = compute_h_statistic_pairwise(model, X_train, feature_j, feature_k)
        pairwise_H[(feature_j, feature_k)] = H2jk

    # Compute Total H-statistics
    logger.info("Computing total H-statistics...")
    for feature_j in tqdm(relevant_features, desc="Total H-statistics"):
        H2j = compute_h_statistic_total(model, X_train, feature_j)
        total_H[feature_j] = H2j

    # Save the statistics
    save_h_statistics(pairwise_H, total_H, save_dir)

    # Summary
    logger.info("H-statistic computation completed.")
    logger.info("Pairwise H-statistics indicate the strength of interaction between feature pairs.")
    logger.info("Total H-statistics indicate the overall interaction strength of each feature with all others.")

def main():
    """
    Main function to orchestrate the computation of H-statistics for all models.
    """
    try:
        # Create explanation directories
        directories = create_directories(EXPLANATIONS_DIR, MODEL_NAMES)

        # Load data and models
        X_train = load_data(DATA_PATH)
        models = load_models(MODEL_PATHS)

        # Compute and save H-statistics for each model
        for model_name, model in models.items():
            save_dir = directories[model_name]
            compute_and_save_h_statistics(model, model_name, X_train, save_dir)

        logger.info("All H-statistics have been computed and saved successfully.")

    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()