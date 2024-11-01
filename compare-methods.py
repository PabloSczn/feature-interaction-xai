import os
import sys
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
DATASET_NAME = 'bike-sharing'  # Change to 'friedman1' or 'bike-sharing' as needed
MODELS = ['xgb', 'rf']  # List of models to process
OUTPUT_DIR = './comparison_results'  # Base directory to save the comparison results
TOP_N = 10  # Number of top interactions to consider

# Feature Aliases for readability in plots and reports
FEATURE_ALIASES = {
    'weather_situation_mist + cloudy, mist + broken clouds, mist + few clouds, mist': 'weather_mist_cloudy+few_broken_clouds',
    'weather_situation_light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds': 'weather_light_snow_rain+thunderstorm+scatteredclouds',
}

def create_output_directories(base_dir, dataset_name, models):
    """
    Create directories for saving comparison results for each model within the dataset.

    Args:
        base_dir (str): Base directory path.
        dataset_name (str): Name of the dataset.
        models (list): List of model names.

    Returns:
        dict: Mapping of model names to their respective directories.
    """
    directories = {}
    try:
        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        logger.debug(f"Directory ensured: {dataset_dir}")
        for model in models:
            model_dir = os.path.join(dataset_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            directories[model] = model_dir
            logger.debug(f"Directory ensured: {model_dir}")
        return directories
    except Exception as e:
        logger.error(f"Failed to create directories in {base_dir}: {e}")
        raise

def load_ale_interactions(model_name):
    """
    Load 2D ALE interaction metrics for a given model.

    Args:
        model_name (str): Name of the model.

    Returns:
        pd.DataFrame: DataFrame containing ALE interaction metrics.
    """
    try:
        file_path = f'./explanations/ale/{DATASET_NAME}/{model_name}/Interaction_Metrics/ale_interaction_metrics_2D.csv'
        if not os.path.exists(file_path):
            logger.warning(f"ALE interaction metrics file not found for model '{model_name}' at '{file_path}'.")
            return pd.DataFrame()
        ale_df = pd.read_csv(file_path)
        logger.info(f"ALE interaction metrics loaded for model '{model_name}'.")
        return ale_df
    except Exception as e:
        logger.error(f"Failed to load ALE interaction metrics for model '{model_name}': {e}")
        return pd.DataFrame()

def load_hstat_interactions(model_name):
    """
    Load pairwise H-statistics for a given model.

    Args:
        model_name (str): Name of the model.

    Returns:
        pd.DataFrame: DataFrame containing H-statistics.
    """
    try:
        file_path = f'./explanations/h-statistic/{DATASET_NAME}/{model_name}/pairwise_H_statistic.csv'
        if not os.path.exists(file_path):
            logger.warning(f"H-statistic file not found for model '{model_name}' at '{file_path}'.")
            return pd.DataFrame()
        hstat_df = pd.read_csv(file_path)
        logger.info(f"H-statistics loaded for model '{model_name}'.")
        return hstat_df
    except Exception as e:
        logger.error(f"Failed to load H-statistics for model '{model_name}': {e}")
        return pd.DataFrame()

def load_shap_interactions(model_name):
    """
    Load SHAP interaction values for a given model.

    Args:
        model_name (str): Name of the model.

    Returns:
        pd.DataFrame: DataFrame containing SHAP interaction values.
    """
    try:
        file_path = f'./explanations/shap-interaction/{DATASET_NAME}/{model_name}/{model_name}_shap_interaction_values.csv'
        if not os.path.exists(file_path):
            logger.warning(f"SHAP interaction values file not found for model '{model_name}' at '{file_path}'.")
            return pd.DataFrame()
        shap_df = pd.read_csv(file_path)
        logger.info(f"SHAP interaction values loaded for model '{model_name}'.")
        return shap_df
    except Exception as e:
        logger.error(f"Failed to load SHAP interaction values for model '{model_name}': {e}")
        return pd.DataFrame()

def merge_interactions(ale_df, hstat_df, shap_df):
    """
    Merge interaction data from ALE, H-statistic, and SHAP methods.

    Args:
        ale_df (pd.DataFrame): DataFrame containing ALE interaction metrics.
        hstat_df (pd.DataFrame): DataFrame containing H-statistics.
        shap_df (pd.DataFrame): DataFrame containing SHAP interaction values.

    Returns:
        pd.DataFrame: Merged DataFrame containing interaction data from all methods.
    """
    try:
        # Standardise column names
        if not ale_df.empty:
            ale_df = ale_df.rename(columns={'Feature 1': 'Feature1', 'Feature 2': 'Feature2', 'ALE Interaction Strength': 'ALE'})
        if not hstat_df.empty:
            hstat_df = hstat_df.rename(columns={'Feature 1': 'Feature1', 'Feature 2': 'Feature2', 'H2jk (%)': 'Hstat'})
            # Remove '%' from Hstat and convert to float if necessary
            if hstat_df['Hstat'].dtype == object:
                hstat_df['Hstat'] = hstat_df['Hstat'].str.rstrip('%').astype(float)
        if not shap_df.empty:
            shap_df = shap_df.rename(columns={'Feature_1': 'Feature1', 'Feature_2': 'Feature2', 'Interaction_Value': 'SHAP'})
        # Merge the DataFrames
        merged_df = pd.DataFrame()
        if not ale_df.empty:
            merged_df = ale_df.copy()
        if not hstat_df.empty:
            if merged_df.empty:
                merged_df = hstat_df.copy()
            else:
                merged_df = pd.merge(merged_df, hstat_df, on=['Feature1', 'Feature2'], how='outer')
        if not shap_df.empty:
            if merged_df.empty:
                merged_df = shap_df.copy()
            else:
                merged_df = pd.merge(merged_df, shap_df, on=['Feature1', 'Feature2'], how='outer')
        logger.info("Interaction data merged successfully.")
        
        # Apply feature aliases if the dataset is 'bike-sharing'
        if DATASET_NAME == 'bike-sharing':
            merged_df['Feature1'] = merged_df['Feature1'].replace(FEATURE_ALIASES)
            merged_df['Feature2'] = merged_df['Feature2'].replace(FEATURE_ALIASES)
            logger.info("Feature aliases applied for 'bike-sharing' dataset.")
        
        return merged_df
    except Exception as e:
        logger.error(f"Failed to merge interaction data: {e}")
        return pd.DataFrame()

def normalise_interactions(df):
    """
    Normalise the interaction strengths from different methods to [0, 1] range.

    Args:
        df (pd.DataFrame): DataFrame containing interaction strengths.

    Returns:
        pd.DataFrame: DataFrame with normalised interaction strengths.
    """
    try:
        methods = ['ALE', 'Hstat', 'SHAP']
        for method in methods:
            if method in df.columns:
                min_value = df[method].min()
                max_value = df[method].max()
                if pd.isna(min_value) or pd.isna(max_value):
                    df[method + '_norm'] = np.nan
                    logger.warning(f"Cannot normalise '{method}' due to NaN values.")
                elif max_value - min_value == 0:
                    df[method + '_norm'] = 0.0
                    logger.warning(f"No variation in '{method}' values. All normalised values set to 0.")
                else:
                    df[method + '_norm'] = (df[method] - min_value) / (max_value - min_value)
                    logger.debug(f"Normalised '{method}' successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to normalise interaction strengths: {e}")
        return df

def interpret_interactions(df, model_name, save_dir):
    """
    Interpret the interactions by normalising and ranking them.

    Args:
        df (pd.DataFrame): DataFrame containing merged interaction data.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the interpreted interactions.

    Returns:
        pd.DataFrame: DataFrame with interpreted interactions.
    """
    try:
        # Normalise the interaction strengths
        df = normalise_interactions(df)
        
        # Compute average normalised interaction strength
        norm_columns = [col for col in df.columns if col.endswith('_norm')]
        if norm_columns:
            df['Average_norm'] = df[norm_columns].mean(axis=1, skipna=True)
            logger.debug("Computed average normalised interaction strength.")
        else:
            df['Average_norm'] = np.nan
            logger.warning("No normalised interaction columns found to compute average.")
        
        # Rank interactions
        if 'ALE_norm' in df.columns:
            df['Rank_ALE'] = df['ALE_norm'].rank(ascending=False, method='min')
        else:
            df['Rank_ALE'] = np.nan
            logger.warning("'ALE_norm' column not found for ranking.")
        
        if 'Hstat_norm' in df.columns:
            df['Rank_Hstat'] = df['Hstat_norm'].rank(ascending=False, method='min')
        else:
            df['Rank_Hstat'] = np.nan
            logger.warning("'Hstat_norm' column not found for ranking.")
        
        if 'SHAP_norm' in df.columns:
            df['Rank_SHAP'] = df['SHAP_norm'].rank(ascending=False, method='min')
        else:
            df['Rank_SHAP'] = np.nan
            logger.warning("'SHAP_norm' column not found for ranking.")
        
        if 'Average_norm' in df.columns:
            df['Rank_Average'] = df['Average_norm'].rank(ascending=False, method='min')
        else:
            df['Rank_Average'] = np.nan
            logger.warning("'Average_norm' column not found for ranking.")
        
        # Save the interpreted interactions
        output_path = os.path.join(save_dir, f'{model_name}_interpreted_interactions.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Interpreted interactions saved at '{output_path}'.")
        return df
    except Exception as e:
        logger.error(f"Failed to interpret interactions for model '{model_name}': {e}")
        return pd.DataFrame()

def plot_heatmap(df, model_name, save_dir):
    """
    Generate heatmaps of normalised interaction strengths.

    Args:
        df (pd.DataFrame): DataFrame containing interpreted interactions.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the heatmaps.

    Returns:
        None
    """
    try:
        methods = ['ALE_norm', 'Hstat_norm', 'SHAP_norm']
        for method in methods:
            if method in df.columns:
                pivot_df = df.pivot(index='Feature1', columns='Feature2', values=method)
                # Since interactions are symmetric, we can fill the other half
                pivot_df = pivot_df.combine_first(pivot_df.T)
                # Replace NaN with 0 for visualisation purposes
                pivot_df_filled = pivot_df.fillna(0)
                
                plt.figure(figsize=(14, 12))  # Increased figure size
                sns.heatmap(
                    pivot_df_filled, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='viridis', 
                    square=True,
                    annot_kws={"size": 8}  # Smaller annotation font size
                )
                plt.title(f'{method} Interaction Heatmap for {model_name.upper()}', fontsize=16)
                
                # Rotate x and y labels for better readability
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(rotation=0, fontsize=10)
                
                plt.tight_layout()
                output_path = os.path.join(save_dir, f'{model_name}_{method}_heatmap.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
                logger.info(f"Heatmap saved at '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to generate heatmaps for model '{model_name}': {e}")

def plot_top_interactions(df, model_name, save_dir, top_n=TOP_N):
    """
    Plot the top N interactions according to each method.

    Args:
        df (pd.DataFrame): DataFrame containing interpreted interactions.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the plots.
        top_n (int): Number of top interactions to plot.

    Returns:
        None
    """
    try:
        methods = ['ALE_norm', 'Hstat_norm', 'SHAP_norm']
        for method in methods:
            if method in df.columns:
                top_df = df.nlargest(top_n, method).copy()
                if top_df.empty:
                    logger.warning(f"No data available to plot top {top_n} interactions for method '{method}'.")
                    continue
                top_df['Feature_Pair'] = top_df['Feature1'] + ' & ' + top_df['Feature2']
                
                plt.figure(figsize=(14, 10))  # Increased figure size
                sns.barplot(
                    x=method,
                    y='Feature_Pair',
                    data=top_df,
                    palette='viridis'
                )
                plt.title(f'Top {top_n} Interactions by {method} for {model_name.upper()}', fontsize=16)
                plt.xlabel('Normalised Interaction Strength', fontsize=12)
                plt.ylabel('Feature Pair', fontsize=12)
                
                # Rotate y-axis labels if they are too long
                plt.yticks(rotation=0, fontsize=10)
                plt.xticks(fontsize=10)
                
                plt.tight_layout()
                output_path = os.path.join(save_dir, f'{model_name}_top_{top_n}_{method}_interactions.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
                logger.info(f"Top {top_n} interactions plot saved at '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to plot top interactions for model '{model_name}': {e}")

def plot_scatter_comparison(df, model_name, save_dir):
    """
    Plot scatter plots comparing normalised interaction strengths between methods.

    Args:
        df (pd.DataFrame): DataFrame containing interpreted interactions.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the plots.

    Returns:
        None
    """
    try:
        method_pairs = [('ALE_norm', 'Hstat_norm'), ('ALE_norm', 'SHAP_norm'), ('Hstat_norm', 'SHAP_norm')]
        for method_x, method_y in method_pairs:
            if method_x in df.columns and method_y in df.columns:
                scatter_df = df.dropna(subset=[method_x, method_y])
                if scatter_df.empty:
                    logger.warning(f"No overlapping data to plot '{method_x}' vs '{method_y}' for model '{model_name}'.")
                    continue
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=method_x, y=method_y, data=scatter_df)
                plt.title(f'{method_x} vs {method_y} for {model_name.upper()}')
                plt.xlabel(method_x)
                plt.ylabel(method_y)
                plt.tight_layout()
                output_path = os.path.join(save_dir, f'{model_name}_{method_x}_vs_{method_y}.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
                logger.info(f"Scatter plot saved at '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to plot scatter comparisons for model '{model_name}': {e}")

def generate_report(df, model_name, save_dir):
    """
    Generate a textual report interpreting the interactions.

    Args:
        df (pd.DataFrame): DataFrame containing interpreted interactions.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the report.

    Returns:
        None
    """
    try:
        report_path = os.path.join(save_dir, f'{model_name}_interaction_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Interaction Report for Model: {model_name.upper()}\n")
            f.write("="*50 + "\n\n")
            f.write("Top Interactions by Average Normalised Interaction Strength:\n")
            top_df = df.sort_values(by='Average_norm', ascending=False).head(TOP_N)
            for idx, row in top_df.iterrows():
                feature_pair = f"{row['Feature1']} & {row['Feature2']}"
                f.write(f"{idx+1}. {feature_pair}: Average Norm = {row['Average_norm']:.2f}\n")
                if 'ALE' in df.columns and not pd.isna(row['ALE']):
                    f.write(f"   - ALE: {row['ALE']:.4f}, Norm: {row['ALE_norm']:.2f}, Rank: {int(row['Rank_ALE']) if not pd.isna(row['Rank_ALE']) else 'N/A'}\n")
                else:
                    f.write(f"   - ALE: N/A\n")
                if 'Hstat' in df.columns and not pd.isna(row['Hstat']):
                    f.write(f"   - H-statistic: {row['Hstat']:.2f}%, Norm: {row['Hstat_norm']:.2f}, Rank: {int(row['Rank_Hstat']) if not pd.isna(row['Rank_Hstat']) else 'N/A'}\n")
                else:
                    f.write(f"   - H-statistic: N/A\n")
                if 'SHAP' in df.columns and not pd.isna(row['SHAP']):
                    f.write(f"   - SHAP: {row['SHAP']:.4f}, Norm: {row['SHAP_norm']:.2f}, Rank: {int(row['Rank_SHAP']) if not pd.isna(row['Rank_SHAP']) else 'N/A'}\n")
                else:
                    f.write(f"   - SHAP: N/A\n")
                f.write("\n")
            f.write("\n")
            f.write("Conclusion:\n")
            f.write("The above feature pairs are considered to be the most interacting according to the methods.\n")
        logger.info(f"Interaction report saved at '{report_path}'.")
    except Exception as e:
        logger.error(f"Failed to generate interaction report for model '{model_name}': {e}")

def main():
    """
    Main function to orchestrate the comparison of XAI methods.
    """
    try:
        logger.info("Starting XAI methods comparison script...")

        # Create output directories with dataset name
        output_dirs = create_output_directories(OUTPUT_DIR, DATASET_NAME, MODELS)

        for model in MODELS:
            logger.info(f"Processing model '{model.upper()}'...")

            # Load interactions
            ale_df = load_ale_interactions(model)
            hstat_df = load_hstat_interactions(model)
            shap_df = load_shap_interactions(model)

            if ale_df.empty and hstat_df.empty and shap_df.empty:
                logger.warning(f"No interaction data found for model '{model}'. Skipping comparison.")
                continue

            # Merge interactions
            merged_df = merge_interactions(ale_df, hstat_df, shap_df)

            if merged_df.empty:
                logger.warning(f"Merged DataFrame is empty for model '{model}'. Skipping further processing.")
                continue

            # Interpret interactions (normalise and save)
            merged_df = interpret_interactions(merged_df, model, output_dirs[model])

            if merged_df.empty:
                logger.warning(f"Interpreted DataFrame is empty for model '{model}'. Skipping visualisation.")
                continue

            # Generate visualisations
            plot_heatmap(merged_df, model, output_dirs[model])
            plot_top_interactions(merged_df, model, output_dirs[model], TOP_N)
            plot_scatter_comparison(merged_df, model, output_dirs[model])

            # Generate textual report
            generate_report(merged_df, model, output_dirs[model])

        logger.info("XAI methods comparison script completed successfully.")

    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()