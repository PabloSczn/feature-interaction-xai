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
DATASET_NAME = 'friedman1'  # Change to 'friedman1' or 'bike-sharing' as needed

# Define paths based on the dataset name
if DATASET_NAME == 'friedman1':
    DATA_PATH = './data/friedman_X_train.csv'
    MODEL_BASE_PATH = './models/friedman'
    EXPLANATIONS_DIR = './explanations/ale/friedman1'
elif DATASET_NAME == 'bike-sharing':
    DATA_PATH = './data/bike_sharing_processed.csv'
    MODEL_BASE_PATH = './models/bike-sharing'
    EXPLANATIONS_DIR = './explanations/ale/bike-sharing'
else:
    logger.error(f"Unsupported DATASET_NAME: {DATASET_NAME}. Supported datasets are 'friedman1' and 'bike-sharing'.")
    sys.exit(1)

# Define model paths based on the dataset
MODEL_PATHS = {
    'xgb': os.path.join(MODEL_BASE_PATH, 'xgb_model.pkl'),
    'rf': os.path.join(MODEL_BASE_PATH, 'rf_model.pkl')
}

MODEL_NAMES = ['xgb', 'rf']
GRID_SIZE = 50
DPI = 700

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
                'Interaction_Metrics': os.path.join(model_dir, 'Interaction_Metrics')
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
    Load the dataset from the specified CSV file and preprocess it based on the dataset name.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and preprocessed feature dataset.
    """
    try:
        logger.info(f"Loading dataset from {data_path}...")
        data = pd.read_csv(data_path)
        logger.info(f"Dataset loaded with shape {data.shape}.")

        if DATASET_NAME == 'bike-sharing':
            logger.info("Preprocessing data for 'bike-sharing' dataset...")
            # Drop the target variable
            if 'total_count' in data.columns:
                data = data.drop('total_count', axis=1)
                logger.debug("'total_count' column dropped from features.")

            # One-hot encode categorical features
            categorical_features = ['season', 'weather_situation']
            logger.info(f"Applying one-hot encoding to categorical features: {categorical_features}")
            data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
            logger.info(f"One-hot encoding applied. New shape: {data.shape}")
        elif DATASET_NAME == 'friedman1':
            logger.info("No preprocessing required for 'friedman1' dataset.")
            pass
        else:
            logger.error(f"Unsupported DATASET_NAME: {DATASET_NAME}.")
            sys.exit(1)

        logger.info(f"Preprocessed dataset shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in the file {data_path}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading or preprocessing data: {e}")
        sys.exit(1)

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

def format_axis_ticks(ax, feature1=None, feature2=None):
    """
    Format the axis ticks for better readability.

    Args:
        ax (matplotlib.axes.Axes): The axes to format.
        feature1 (str, optional): The first feature name (for 2D plots).
        feature2 (str, optional): The second feature name (for 2D plots).

    Returns:
        None
    """
    try:
        # Format x-axis ticks
        ticks = ax.get_xticks()
        tick_labels = [f"{tick:.2f}" for tick in ticks]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Format y-axis ticks
        ticks = ax.get_yticks()
        tick_labels = [f"{tick:.2f}" for tick in ticks]
        ax.set_yticklabels(tick_labels, rotation=0)
    except Exception as e:
        logger.warning(f"Failed to format axis ticks: {e}")

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
        if interaction_metrics_df_filtered.empty:
            logger.warning("No 2D interaction metrics available to generate heatmap.")
            return

        # Pivot the DataFrame to create a matrix
        pivot_df = interaction_metrics_df_filtered.pivot(index='Feature 1', columns='Feature 2', values='ALE Interaction Strength')
        # Sort the features for better visualisation
        pivot_df = pivot_df.sort_index().sort_index(axis=1)
        # Create a symmetric matrix by combining with its transpose
        pivot_df = pivot_df.combine_first(pivot_df.T)
        # Replace diagonal with NaN to avoid self-interaction
        for feature in pivot_df.index:
            pivot_df.at[feature, feature] = np.nan

        # Shorten feature names for readability
        # Define a mapping from long feature names to shorter aliases
        feature_aliases = {
            'weather_situation_mist + cloudy, mist + broken clouds, mist + few clouds, mist': 'weather_mist_cloudy+few_broken_clouds',
            'weather_situation_light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds': 'weather_light_snow_rain+thunderstorm+scatteredclouds',
        }

        # Apply the mapping to the pivot DataFrame
        pivot_df.rename(index=feature_aliases, columns=feature_aliases, inplace=True)

        plt.figure(figsize=(20, 18))  # Increased figure size for better readability
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap='viridis',
            square=True,
            cbar_kws={"shrink": .8},
            annot_kws={"size": 8},  # Smaller annotation font size
            linewidths=.5,  # Add lines between cells for clarity
            linecolor='grey'
        )
        plt.title(f"ALE Interaction Strength Heatmap for {model_name.upper()}", fontsize=16)
        plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels vertically
        plt.yticks(rotation=0, fontsize=10)   # Keep y-axis labels horizontal
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f'ale_interaction_heatmap_{model_name}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Interaction heatmap saved: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate interaction heatmap: {e}")
        raise

def split_interaction_metrics(interaction_metrics_df, save_dir):
    """
    Split interaction metrics into 1D and 2D CSV files.

    Args:
        interaction_metrics_df (pd.DataFrame): DataFrame containing interaction metrics.
        save_dir (str): Directory to save the split CSV files.

    Returns:
        None
    """
    try:
        # 1D ALE interactions where Feature 2 is 'All'
        interaction_metrics_1D = interaction_metrics_df[interaction_metrics_df['Feature 2'] == 'All']
        if not interaction_metrics_1D.empty:
            interaction_metrics_1D_path = os.path.join(save_dir, 'ale_interaction_metrics_1D.csv')
            interaction_metrics_1D.to_csv(interaction_metrics_1D_path, index=False)
            logger.info(f"ALE Interaction Metrics (1D) saved to {interaction_metrics_1D_path}")
        else:
            logger.warning("No 1D ALE interaction metrics to save.")

        # 2D ALE interactions where Feature 2 is not 'All'
        interaction_metrics_2D = interaction_metrics_df[interaction_metrics_df['Feature 2'] != 'All']
        if not interaction_metrics_2D.empty:
            interaction_metrics_2D_path = os.path.join(save_dir, 'ale_interaction_metrics_2D.csv')
            interaction_metrics_2D.to_csv(interaction_metrics_2D_path, index=False)
            logger.info(f"ALE Interaction Metrics (2D) saved to {interaction_metrics_2D_path}")
        else:
            logger.warning("No 2D ALE interaction metrics to save.")
    except Exception as e:
        logger.error(f"Failed to split interaction metrics: {e}")
        raise

def format_2d_ale_plot(ax, feature1, feature2):
    """
    Format the 2D ALE plot axes for better readability.

    Args:
        ax (matplotlib.axes.Axes): The axes to format.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.

    Returns:
        None
    """
    try:
        # Round the tick labels to 2 decimal places
        ticks = ax.get_xticks()
        tick_labels = [f"{tick:.2f}" for tick in ticks]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        ticks = ax.get_yticks()
        tick_labels = [f"{tick:.2f}" for tick in ticks]
        ax.set_yticklabels(tick_labels, rotation=0)
    except Exception as e:
        logger.warning(f"Failed to format 2D ALE plot axes for features '{feature1}' and '{feature2}': {e}")

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

                # Format axis ticks for better readability
                format_2d_ale_plot(ax, feature1, feature2)

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
        # Split interaction metrics into 1D and 2D
        split_interaction_metrics(interaction_metrics_df, save_dirs['Interaction_Metrics'])

        # Generate interaction heatmap based on 2D interaction metrics
        interaction_metrics_2D = interaction_metrics_df[interaction_metrics_df['Feature 2'] != 'All']
        if not interaction_metrics_2D.empty:
            generate_interaction_heatmap(interaction_metrics_2D, save_dirs['Interaction_Metrics'], model_name)
        else:
            logger.warning("No 2D ALE interaction metrics available to generate heatmap.")
    else:
        logger.warning("No ALE interaction metrics were computed.")

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
    - 2D ALE Plot: Visualises interactions between two features and how their joint effects contribute to the model’s decision-making.
    - ALE Interaction Metrics: Numerical values representing the strength of interactions between feature pairs.
    - Interaction Heatmap: Visual representation of interaction strengths for easy comparison.
    - Interaction Metrics CSV: Separated into 1D and 2D for better clarity.
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