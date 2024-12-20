import os
import sys
import logging

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
DATASET_NAME = 'friedman1'  # Change to 'friedman1' or 'bike-sharing' as needed

# Dynamic path configurations based on DATASET_NAME
if DATASET_NAME == 'friedman1':
    DATA_PATH = './data/friedman_X_train.csv'
    MODEL_PATHS = {
        'xgb': './models/friedman/xgb_model.pkl',
        'rf': './models/friedman/rf_model.pkl'
    }
    EXPLANATIONS_DIR = "explanations/shap-interaction/friedman1"
elif DATASET_NAME == 'bike-sharing':
    DATA_PATH = './data/bike_sharing_processed.csv'
    MODEL_PATHS = {
        'xgb': './models/bike-sharing/xgb_model.pkl',
        'rf': './models/bike-sharing/rf_model.pkl'
    }
    EXPLANATIONS_DIR = "explanations/shap-interaction/bike-sharing"
else:
    logger.error(f"Unsupported DATASET_NAME: {DATASET_NAME}. Choose 'friedman1' or 'bike-sharing'.")
    sys.exit(1)

MODEL_NAMES = ['xgb', 'rf']
TOP_N_INTERACTIONS = 10

def create_directories(base_dir, model_names):
    """
    Create directories for saving SHAP interaction explanations for each model.

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
        feature_names = X_train.columns.tolist()
        logger.info(f"Dataset loaded with {X_train.shape[0]} instances and {X_train.shape[1]} features.")
        return X_train, feature_names
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in the file {data_path}.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

def preprocess_data(X, dataset_name):
    """
    Preprocess the data by applying necessary transformations.

    Args:
        X (pd.DataFrame): Raw feature dataset.
        dataset_name (str): Name of the dataset ('friedman1' or 'bike-sharing').

    Returns:
        pd.DataFrame: Preprocessed feature dataset.
    """
    if dataset_name == 'bike-sharing':
        logger.info("Applying one-hot encoding to categorical features: 'season' and 'weather_situation'.")
        categorical_features = ['season', 'weather_situation']
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        logger.info(f"One-hot encoding applied. New shape: {X.shape}")
    return X

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

def get_model_features(model, model_name):
    """
    Get the feature names expected by the model.

    Args:
        model: Trained model.
        model_name (str): 'xgb' or 'rf'

    Returns:
        list: Feature names expected by the model.
    """
    if model_name == 'xgb':
        return model.get_booster().feature_names
    elif model_name == 'rf':
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        else:
            logger.error(f"Random Forest model '{model_name}' does not have 'feature_names_in_' attribute.")
            raise AttributeError("Model does not have 'feature_names_in_'")
    else:
        logger.error(f"Unsupported model type: {model_name}")
        raise ValueError("Unsupported model type")

def compute_shap_interactions(model, X):
    """
    Compute SHAP interaction values for a given model and dataset.

    Args:
        model: The pre-trained machine learning model.
        X (pd.DataFrame): The feature matrix.

    Returns:
        np.ndarray: SHAP interaction values.
    """
    try:
        logger.info("Initialising SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        logger.info("Computing SHAP interaction values...")
        shap_interactions = explainer.shap_interaction_values(X)
        logger.info("SHAP interaction values computed successfully.")
        return shap_interactions
    except Exception as e:
        logger.error(f"Failed to compute SHAP interaction values: {e}")
        raise

def aggregate_interactions(shap_interactions, feature_names):
    """
    Aggregate SHAP interaction values to identify the most significant feature interactions.

    Args:
        shap_interactions (np.ndarray): SHAP interaction values.
        feature_names (list): List of feature names.

    Returns:
        tuple: (interaction_df, interaction_matrix)
            - interaction_df (pd.DataFrame): DataFrame containing feature pairs and their mean interaction effects.
            - interaction_matrix (np.ndarray): Matrix of interaction values.
    """
    try:
        logger.info("Aggregating SHAP interaction values...")
        num_features = len(feature_names)
        interaction_matrix = np.zeros((num_features, num_features))

        # Compute mean absolute interaction values across all instances
        mean_interactions = np.mean(np.abs(shap_interactions), axis=0)

        # Fill the interaction_matrix
        for i in range(num_features):
            for j in range(num_features):
                interaction_matrix[i, j] = mean_interactions[i, j]

        # Create a DataFrame for easy manipulation
        interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

        # Remove self-interactions by setting diagonal to NaN
        np.fill_diagonal(interaction_matrix, np.nan)

        # Melt the DataFrame to have feature pairs and their interaction values
        interaction_df = interaction_df.reset_index().melt(id_vars='index', var_name='Feature_2', value_name='Interaction_Value')
        interaction_df.rename(columns={'index': 'Feature_1'}, inplace=True)

        # Remove duplicate pairs and self-interactions
        interaction_df = interaction_df[interaction_df['Feature_1'] < interaction_df['Feature_2']]

        # Sort by Interaction_Value descending
        interaction_df = interaction_df.sort_values(by='Interaction_Value', ascending=False).reset_index(drop=True)

        logger.info("Aggregation of SHAP interaction values completed.")
        return interaction_df, interaction_matrix
    except Exception as e:
        logger.error(f"Failed to aggregate SHAP interaction values: {e}")
        raise

def save_interaction_explanations(model_name, interaction_df, interaction_matrix, output_dir, feature_names, top_n=TOP_N_INTERACTIONS):
    """
    Save the interaction explanations including CSV and heatmap visualisation.

    Args:
        model_name (str): Name of the model (e.g., 'xgb', 'rf').
        interaction_df (pd.DataFrame): DataFrame containing feature pairs and their mean interaction effects.
        interaction_matrix (np.ndarray): Matrix of interaction values.
        output_dir (str): Directory to save the explanations.
        feature_names (list): List of feature names.
        top_n (int): Number of top interactions to save/display.
    """
    try:
        logger.info(f"Saving SHAP interaction explanations for model '{model_name.upper()}'...")

        # Save the aggregated interactions as CSV
        csv_path = os.path.join(output_dir, f"{model_name}_shap_interaction_values.csv")
        interaction_df.to_csv(csv_path, index=False)
        logger.info(f"Saved SHAP interaction values CSV at {csv_path}")

        # Save the top N interactions as a separate CSV for easy reference
        top_interactions = interaction_df.head(top_n)
        top_csv_path = os.path.join(output_dir, f"{model_name}_top_{top_n}_shap_interactions.csv")
        top_interactions.to_csv(top_csv_path, index=False)
        logger.info(f"Saved top {top_n} SHAP interactions CSV at {top_csv_path}")

        # Generate and save a heatmap of interaction values
        logger.info("Generating SHAP interaction heatmap...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_matrix, xticklabels=feature_names, yticklabels=feature_names, 
                    cmap='viridis', annot=False, fmt=".2f")
        plt.title(f"SHAP Interaction Values Heatmap for {model_name.upper()} ({DATASET_NAME})")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f"{model_name}_shap_interactions_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP interaction heatmap at {heatmap_path}")

    except Exception as e:
        logger.error(f"Failed to save SHAP interaction explanations for model '{model_name}': {e}")
        raise

def compute_and_save_shap_interactions(model, model_name, X_train, feature_names, save_dir, dataset_name):
    """
    Compute and save SHAP interaction explanations for a given model.

    Args:
        model: Trained machine learning model.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Training feature dataset.
        feature_names (list): List of feature names.
        save_dir (str): Directory to save the explanations.
        dataset_name (str): Name of the dataset.
    """
    try:
        logger.info(f"Processing SHAP interaction values for model '{model_name.upper()}'...")

        # Get model's expected feature names
        model_features = get_model_features(model, model_name)

        # Align X_train to model's feature names
        X_train_aligned = X_train.reindex(columns=model_features, fill_value=0)

        # Verify alignment
        if X_train_aligned.shape[1] != len(model_features):
            logger.error("Mismatch in the number of features after alignment.")
            raise ValueError("Feature alignment failed.")

        # Compute SHAP interaction values
        shap_interactions = compute_shap_interactions(model, X_train_aligned)

        # Aggregate interactions
        interaction_df, interaction_matrix = aggregate_interactions(shap_interactions, model_features)

        # Save explanations
        save_interaction_explanations(model_name, interaction_df, interaction_matrix, save_dir, model_features)

        logger.info(f"SHAP interaction explanations for model '{model_name.upper()}' have been saved successfully.")
    except Exception as e:
        logger.error(f"Failed to compute and save SHAP interactions for model '{model_name}': {e}")
        raise

def main():
    """
    Main function to orchestrate the computation of SHAP interaction values for all models.
    """
    try:
        logger.info(f"Starting SHAP interaction values computation for dataset '{DATASET_NAME}'...")

        # Create explanation directories
        directories = create_directories(EXPLANATIONS_DIR, MODEL_NAMES)

        # Load data
        X_train, feature_names = load_data(DATA_PATH)

        # Preprocess data if necessary
        X_train = preprocess_data(X_train, DATASET_NAME)

        # Update feature names after preprocessing
        feature_names = X_train.columns.tolist()

        # Load models
        models = load_models(MODEL_PATHS)

        # Compute and save SHAP interaction explanations for each model
        for model_name, model in models.items():
            save_dir = directories[model_name]
            compute_and_save_shap_interactions(model, model_name, X_train, feature_names, save_dir, DATASET_NAME)

        logger.info("All SHAP interaction explanations have been computed and saved successfully.")

    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()