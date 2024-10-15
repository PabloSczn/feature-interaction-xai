import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PyALE import ale
import logging
from itertools import combinations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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


def load_model(model_path):
    """
    Load a pre-trained model from the given path.

    Args:
        model_path (str): Path to the model file.
    Returns:
        model: Loaded model object.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        logger.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {model.__class__.__name__}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def create_directories(base_dir, model_names):
    """
    Create directories to save ALE explanations for each model.

    Args:
        base_dir (str): Base directory to save explanations.
        model_names (list): List of model names.
    Returns:
        dict: Dictionary mapping model names to their respective directories.
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


def generate_ale_explanations(model, model_name, X_train, save_dir):
    """
    Generate and save ALE explanations (1D and 2D) for the given model.

    Args:
        model: The trained model to explain.
        model_name (str): Name of the model (e.g., 'xgb', 'rf').
        X_train (pd.DataFrame): Feature dataset.
        save_dir (str): Directory to save the explanations.
    """
    logger.info(f"Generating ALE explanations for {model_name}...")

    # 1D ALE: Analyse individual feature effects
    for feature in X_train.columns:
        logger.debug(f"Generating 1D ALE for feature: {feature}")
        try:
            ale_eff = ale(X=X_train, model=model, feature=[feature], grid_size=50, include_CI=True)
            plt.figure(figsize=(8, 6), constrained_layout=True)
            ale_eff.plot()

            # Access current Axes
            ax = plt.gca()

            # Customise title and labels
            ax.set_title(f"1D ALE for {feature} ({model_name.upper()})", fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Accumulated Local Effect', fontsize=12)

            # Adjust legend
            if ax.get_legend():
                ax.legend(fontsize=10, loc='upper right')

            # Saving the plot with bbox_inches='tight' to ensure all elements are within the figure
            plot_path = os.path.join(save_dir, f'ale_1d_{feature}.png')
            plt.savefig(plot_path, dpi=700, bbox_inches='tight')
            plt.close()
            logger.debug(f"1D ALE plot saved: {plot_path}")
        except Exception as e:
            logger.warning(f"Skipped 1D ALE for {feature}: {e}")

    # 2D ALE: Analyse interactions between feature pairs
    logger.debug(f"Generating 2D ALE plots for {model_name}...")
    feature_pairs = combinations(X_train.columns, 2)
    for feature1, feature2 in feature_pairs:
        logger.debug(f"Generating 2D ALE for features: {feature1}, {feature2}")
        try:
            ale_eff = ale(X=X_train, model=model, feature=[feature1, feature2], grid_size=50, include_CI=True)
            plt.figure(figsize=(10, 8), constrained_layout=True)
            ale_eff.plot()

            # Access current Axes
            ax = plt.gca()

            # Customise title and labels
            ax.set_title(f"2D ALE for {feature1} & {feature2} ({model_name.upper()})", fontsize=14)
            ax.set_xlabel(feature2, fontsize=12)
            ax.set_ylabel(feature1, fontsize=12)

            # Adjust legend
            if ax.get_legend():
                ax.legend(fontsize=10, loc='upper right')

            # Saving the plot with bbox_inches='tight' to ensure all elements are within the figure
            plot_path = os.path.join(save_dir, f'ale_2d_{feature1}_{feature2}.png')
            plt.savefig(plot_path, dpi=700, bbox_inches='tight')
            plt.close()
            logger.debug(f"2D ALE plot saved: {plot_path}")
        except Exception as e:
            logger.warning(f"Skipped 2D ALE for {feature1} and {feature2}: {e}")


def main():
    """
    Generates ALE plots for both 1D and 2D feature interactions.

    Explanations:
    - 1D ALE plot: Shows the effect of each feature on the model’s predictions, independent of other features.
    - 2D ALE plot: Visualises interactions between two features and how their joint effects contribute to the model’s decision-making.

    These explanations help compare the feature importance and interaction effects across models.

    Plots are saved for comparison purposes.
    """
    # Paths to data and models
    DATA_PATH = './data/friedman_X_train.csv'
    MODEL_PATHS = {
        'xgb': './models/xgb_model.pkl',
        'rf': './models/rf_model.pkl'
    }

    # Directory to save explanations
    EXPLANATIONS_DIR = './explanations/ale/'

    try:
        # Load the dataset
        X_train = load_data(DATA_PATH)

        # Load the pre-trained models
        models = {}
        for name, path in MODEL_PATHS.items():
            models[name] = load_model(path)

        # Create directories for saving explanations
        directories = create_directories(EXPLANATIONS_DIR, models.keys())

        # Generate ALE explanations for each model
        for model_name, model in models.items():
            generate_ale_explanations(model, model_name, X_train, directories[model_name])

        logger.info("All ALE explanations have been generated and saved successfully.")
    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()