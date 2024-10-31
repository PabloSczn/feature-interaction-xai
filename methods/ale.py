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
# RELEVANT_FEATURES will be set dynamically based on the dataset

def create_directories(base_dir, model_names):
    """
    Create a structured directory to save ALE explanations and additional plots for each model.

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
            subdirs = {
                '1D_ALE': os.path.join(model_dir, '1D_ALE'),
                '2D_ALE': os.path.join(model_dir, '2D_ALE'),
                'Interaction_Metrics': os.path.join(model_dir, 'Interaction_Metrics'),
                'Additional_Plots': os.path.join(model_dir, 'Additional_Plots')
            }
            for subdir in subdirs.values():
                os.makedirs(subdir, exist_ok=True)
                logger.debug(f"Directory ensured: {subdir}")
            directories[model_name] = subdirs
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

def generate_correlation_matrix(X_train, save_dir):
    """
    Generate and save a correlation matrix heatmap.

    Args:
        X_train (pd.DataFrame): Feature dataset.
        save_dir (str): Directory to save the correlation matrix plot.
    """
    try:
        logger.info("Generating correlation matrix...")
        corr_matrix = X_train.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Feature Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation matrix saved: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate correlation matrix: {e}")
        raise

def generate_interaction_heatmap(interaction_metrics_df, save_dir, model_name):
    """
    Generate and save an interaction strength heatmap based on ALE interaction metrics.

    Args:
        interaction_metrics_df (pd.DataFrame): DataFrame containing interaction metrics.
        save_dir (str): Directory to save the interaction heatmap plot.
        model_name (str): Name of the model (e.g., 'xgb', 'rf').
    """
    try:
        logger.info("Generating interaction strength heatmap...")
        # Filter out 1D ALE interactions where Feature 2 is 'All'
        interaction_metrics_df_filtered = interaction_metrics_df[interaction_metrics_df['Feature 2'] != 'All']
        # Ensure feature1 != feature2 (if not already)
        interaction_metrics_df_filtered = interaction_metrics_df_filtered[
            interaction_metrics_df_filtered['Feature 1'] != interaction_metrics_df_filtered['Feature 2']
        ]
        # Pivot the DataFrame to create a matrix
        pivot_df = interaction_metrics_df_filtered.pivot(index='Feature 1', columns='Feature 2', values='ALE Interaction Strength')
        # Sort the features for better visualization
        pivot_df = pivot_df.sort_index().sort_index(axis=1)
        # Create a symmetric matrix by combining with its transpose
        pivot_df = pivot_df.combine_first(pivot_df.T)
        # Replace diagonal with NaN to avoid self-interaction
        for feature in pivot_df.index:
            pivot_df.at[feature, feature] = np.nan
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap='viridis', square=True, cbar_kws={"shrink": .8})
        plt.title(f"ALE Interaction Strength Heatmap for {model_name.upper()}", fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'ale_interaction_heatmap_{model_name}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Interaction heatmap saved: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate interaction heatmap: {e}")
        raise

def generate_ale_explanations(model, model_name, X_train, save_dirs, grid_size=GRID_SIZE, dpi=DPI):
    """
    Generate and save ALE explanations (1D and 2D) for the given model,
    along with additional plots to understand feature interactions.

    Args:
        model: Trained model to explain.
        model_name (str): Name of the model (e.g., 'xgb', 'rf').
        X_train (pd.DataFrame): Feature dataset.
        save_dirs (dict): Directories to save the explanations.
        grid_size (int): Number of grid points for ALE.
        dpi (int): Resolution for saved plots.
    """
    logger.info(f"Generating ALE explanations for model: {model_name.upper()}")

    features = X_train.columns.tolist()

    # Generate 1D ALE plots for all features
    interaction_metrics = []
    logger.info("Generating 1D ALE plots for all features...")
    for feature in features:
        logger.debug(f"Generating 1D ALE for feature: {feature}")
        try:
            # Compute 1D ALE
            ale_eff = ale(
                X=X_train, model=model, feature=[feature], grid_size=grid_size, include_CI=True, C=0.95
            )
            
            # Plot 1D ALE
            plt.figure(figsize=(8, 6), constrained_layout=True)
            ale_eff.plot()
            ax = plt.gca()
            ax.set_title(f"1D ALE for {feature} ({model_name.upper()})", fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Accumulated Local Effect', fontsize=12)
            if ax.get_legend():
                ax.legend(fontsize=10, loc='upper right')
            plot_path = os.path.join(save_dirs['1D_ALE'], f'ale_1d_{feature}.png')
            plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            logger.debug(f"1D ALE plot saved: {plot_path}")

            # Compute interaction strength for 1D ALE (range of effect)
            if 'eff' in ale_eff.columns:
                ale_values = ale_eff['eff'].values
                interaction_strength = np.ptp(ale_values)  # Peak-to-peak (max - min)
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

    # Generate 2D ALE plots for all feature pairs
    logger.info("Generating 2D ALE plots and computing interaction metrics...")
    feature_pairs = list(combinations(features, 2))

    for feature1, feature2 in feature_pairs:
        logger.debug(f"Generating 2D ALE for features: {feature1}, {feature2}")
        try:
            # Compute 2D ALE
            ale_eff = ale(
                X=X_train, model=model, feature=[feature1, feature2], grid_size=grid_size, include_CI=True, C=0.95
            )
            
            # Check if ale_eff is a DataFrame
            if isinstance(ale_eff, pd.DataFrame):
                plt.figure(figsize=(10, 8), constrained_layout=True)
                # Use heatmap to plot 2D ALE
                sns.heatmap(ale_eff, cmap='viridis', cbar_kws={"shrink": .8})
                ax = plt.gca()
                ax.set_title(f"2D ALE for {feature1} & {feature2} ({model_name.upper()})", fontsize=14)
                ax.set_xlabel(feature2, fontsize=12)
                ax.set_ylabel(feature1, fontsize=12)

                # Overlay KDE plot for data distribution
                try:
                    sns.kdeplot(x=X_train[feature2], y=X_train[feature1], levels=5, linewidths=1, colors='white', alpha=0.5, ax=ax)
                except Exception as e:
                    logger.warning(f"Could not overlay KDE plot: {e}")

                plot_path = os.path.join(save_dirs['2D_ALE'], f'ale_2d_{feature1}_{feature2}.png')
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
        interaction_metrics_path = os.path.join(save_dirs['Interaction_Metrics'], 'ale_interaction_metrics.csv')
        interaction_metrics_df.to_csv(interaction_metrics_path, index=False)
        logger.info(f"ALE Interaction Metrics saved to {interaction_metrics_path}")

        # Generate interaction heatmap based on interaction metrics
        generate_interaction_heatmap(interaction_metrics_df, save_dirs['Interaction_Metrics'], model_name)

    else:
        logger.warning("No ALE interaction metrics were computed.")

    # Generate correlation matrix and save in Additional_Plots
    generate_correlation_matrix(X_train, save_dirs['Additional_Plots'])

def compute_and_save_ale_explanations(model, model_name, X_train, save_dirs):
    """
    Compute and save both 1D and 2D ALE explanations for a given model,
    along with additional plots to understand feature interactions.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Feature dataset.
        save_dirs (dict): Directories to save the explanations.
    """
    generate_ale_explanations(model, model_name, X_train, save_dirs)
    logger.info(f"ALE explanations for model '{model_name}' have been generated and saved.")

def main():
    """
    Main function to orchestrate the generation of ALE explanations for all models.

    Explanations:
    - 1D ALE Plot: Shows the effect of each feature on the model’s predictions, independent of other features.
    - 2D ALE Plot: Visualizes interactions between two features and how their joint effects contribute to the model’s decision-making.
    - ALE Interaction Metrics: Numerical values representing the strength of interactions between feature pairs.
    - Correlation Matrix: Shows the correlation between all feature pairs to understand underlying relationships.
    - Interaction Heatmap: Visual representation of interaction strengths for easy comparison.
    - Top Interaction Scatter Plots: Scatter plots for the top N interacting feature pairs to provide deeper insights.
    """
    try:
        # Create explanation directories
        directories = create_directories(EXPLANATIONS_DIR, MODEL_NAMES)

        # Load data and models
        X_train = load_data(DATA_PATH)
        models = load_models(MODEL_PATHS)

        # Compute and save ALE explanations for each model
        for model_name, model in models.items():
            save_dirs = directories[model_name]
            compute_and_save_ale_explanations(model, model_name, X_train, save_dirs)

        logger.info("All ALE explanations have been generated and saved successfully.")

    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()