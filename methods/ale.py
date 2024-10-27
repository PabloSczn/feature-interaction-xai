import os
import sys
import logging
from itertools import combinations

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PyALE import ale
import numpy as np
import seaborn as sns

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
EXPLANATIONS_DIR = './explanations/ale'
MODEL_NAMES = ['xgb', 'rf']
GRID_SIZE = 50
DPI = 700
RELEVANT_FEATURES = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']  # First 5 features

def create_directories(base_dir, model_names):
    """
    Create directories to save ALE explanations for each model.

    Args:
        base_dir (str): Base directory to save explanations.
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

def generate_ale_explanations(model, model_name, X_train, save_dir, grid_size=GRID_SIZE, dpi=DPI):
    """
    Generate and save ALE explanations (1D and 2D) for the given model.

    Args:
        model: Trained model to explain.
        model_name (str): Name of the model (e.g., 'xgb', 'rf').
        X_train (pd.DataFrame): Feature dataset.
        save_dir (str): Directory to save the explanations.
        grid_size (int): Number of grid points for ALE.
        dpi (int): Resolution for saved plots.
    """
    logger.info(f"Generating ALE explanations for model: {model_name.upper()}")

    # Initialize list to store interaction metrics
    interaction_metrics = []

    # Generate 1D ALE plots for relevant features
    for feature in RELEVANT_FEATURES:
        logger.debug(f"Generating 1D ALE for feature: {feature}")
        try:
            # Compute 1D ALE
            ale_eff = ale(X=X_train, model=model, feature=[feature], grid_size=grid_size, include_CI=True)
            
            # Plot 1D ALE
            plt.figure(figsize=(8, 6), constrained_layout=True)
            ale_eff.plot()

            ax = plt.gca()
            ax.set_title(f"1D ALE for {feature} ({model_name.upper()})", fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Accumulated Local Effect', fontsize=12)

            if ax.get_legend():
                ax.legend(fontsize=10, loc='upper right')

            plot_path = os.path.join(save_dir, f'ale_1d_{feature}.png')
            plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            logger.debug(f"1D ALE plot saved: {plot_path}")

            # Compute interaction strength for 1D ALE (range of effect)
            if 'eff' in ale_eff.columns:
                ale_values = ale_eff['eff'].values
                interaction_strength = np.ptp(ale_values)
                interaction_metrics.append({
                    'Feature 1': feature,
                    'Feature 2': 'All',
                    'ALE Interaction Strength': interaction_strength
                })
                logger.debug(f"ALE Interaction Strength for {feature}: {interaction_strength:.4f}")
            else:
                logger.warning(f"'eff' column not found in 1D ALE for feature {feature}")

        except Exception as e:
            logger.warning(f"Skipped 1D ALE for feature '{feature}': {e}")

    # Generate 2D ALE plots for relevant feature pairs and compute interaction metrics
    logger.info("Generating 2D ALE plots and computing interaction metrics...")
    feature_pairs = list(combinations(RELEVANT_FEATURES, 2))

    for feature1, feature2 in feature_pairs:
        logger.debug(f"Generating 2D ALE for features: {feature1}, {feature2}")
        try:
            # Compute 2D ALE
            ale_eff = ale(X=X_train, model=model, feature=[feature1, feature2], grid_size=grid_size, include_CI=True)
            
            # Check if ale_eff is a DataFrame
            if isinstance(ale_eff, pd.DataFrame):
                plt.figure(figsize=(10, 8), constrained_layout=True)

                # Use heatmap to plot 2D ALE
                sns.heatmap(ale_eff, cmap='viridis')
                ax = plt.gca()
                ax.set_title(f"2D ALE for {feature1} & {feature2} ({model_name.upper()})", fontsize=14)
                ax.set_xlabel(feature2, fontsize=12)
                ax.set_ylabel(feature1, fontsize=12)

                # Overlay KDE plot for data distribution
                try:
                    sns.kdeplot(x=X_train[feature2], y=X_train[feature1], levels=5, linewidths=1, colors='white', alpha=0.5, ax=ax)
                except Exception as e:
                    logger.warning(f"Could not overlay KDE plot: {e}")

                plot_path = os.path.join(save_dir, f'ale_2d_{feature1}_{feature2}.png')
                plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                logger.debug(f"2D ALE plot saved: {plot_path}")

                # Compute interaction strength based on the range of ALE effects
                ale_values = ale_eff.values.flatten()
                interaction_strength = np.ptp(ale_values)
                interaction_metrics.append({
                    'Feature 1': feature1,
                    'Feature 2': feature2,
                    'ALE Interaction Strength': interaction_strength
                })
                logger.debug(f"ALE Interaction Strength for ({feature1}, {feature2}): {interaction_strength:.4f}")
            else:
                logger.warning(f"Unexpected type for 2D ALE effect: {type(ale_eff)}. Skipping.")
        except Exception as e:
            logger.warning(f"Skipped 2D ALE for features '{feature1}' and '{feature2}': {e}")

    # Save interaction metrics
    if interaction_metrics:
        interaction_metrics_df = pd.DataFrame(interaction_metrics)
        interaction_metrics_path = os.path.join(save_dir, 'ale_interaction_metrics.csv')
        interaction_metrics_df.to_csv(interaction_metrics_path, index=False)
        logger.info(f"ALE Interaction Metrics saved to {interaction_metrics_path}")
    else:
        logger.warning("No ALE interaction metrics were computed.")

def compute_and_save_ale_explanations(model, model_name, X_train, save_dir):
    """
    Compute and save both 1D and 2D ALE explanations for a given model.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Feature dataset.
        save_dir (str): Directory to save the ALE explanations.
    """
    generate_ale_explanations(model, model_name, X_train, save_dir)
    logger.info(f"ALE explanations for model '{model_name}' have been generated and saved.")

def main():
    """
    Main function to orchestrate the generation of ALE explanations for all models.

    Explanations:
    - 1D ALE Plot: Shows the effect of each feature on the model’s predictions, independent of other features.
    - 2D ALE Plot: Visualizes interactions between two features and how their joint effects contribute to the model’s decision-making.
    - ALE Interaction Metrics: Numerical values representing the strength of interactions between feature pairs.

    These explanations help compare feature importance and interaction effects across models.
    """
    try:
        # Create explanation directories
        directories = create_directories(EXPLANATIONS_DIR, MODEL_NAMES)

        # Load data and models
        X_train = load_data(DATA_PATH)
        models = load_models(MODEL_PATHS)

        # Compute and save ALE explanations for each model
        for model_name, model in models.items():
            save_dir = directories[model_name]
            compute_and_save_ale_explanations(model, model_name, X_train, save_dir)

        logger.info("All ALE explanations have been generated and saved successfully.")

    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()