import os
import sys
import logging
import joblib
import pandas as pd
from imodels import RuleFitRegressor

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
DATA_PATH = './data/friedman_X_train.csv'
TARGET_PATH = './data/friedman_y_train.csv'
MODEL_PATHS = {
    'xgb': './models/xgb_model.pkl',
    'rf': './models/rf_model.pkl'
}
EXPLANATIONS_DIR = "explanations/rulefit"
MODEL_NAMES = ['xgb', 'rf']

def create_directories(base_dir, model_names):
    """
    Create directories for saving RuleFit explanations for each model.
    """
    directories = {}
    for model_name in model_names:
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        directories[model_name] = model_dir
        logger.debug(f"Directory ensured: {model_dir}")
    return directories

def load_data(data_path, target_path):
    """
    Load the dataset from the specified CSV files.
    """
    try:
        logger.info(f"Loading dataset from {data_path} and {target_path}...")
        X_train = pd.read_csv(data_path)
        y_train = pd.read_csv(target_path).values.ravel()
        logger.info(f"Dataset loaded with shape {X_train.shape}.")
        return X_train, y_train
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

def load_models(model_paths):
    """
    Load pre-trained models from the specified paths.
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

def fit_rulefit_model(X_train, y_preds):
    """
    Fit a RuleFit model to approximate the predictions of the trained model.
    """
    logger.info("Fitting RuleFit model...")
    rf_model = RuleFitRegressor(random_state=42)
    rf_model.fit(X_train.values, y_preds, feature_names=X_train.columns.tolist())
    logger.info("RuleFit model fitted.")
    return rf_model

def extract_rules(rulefit_model):
    """
    Extract rules from the RuleFit model.
    """
    logger.info("Extracting rules from RuleFit model...")
    # Using the private method to get rules with 'importance' column
    rules = rulefit_model._get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    logger.info(f"Extracted {len(rules)} rules with non-zero coefficients.")
    return rules

def identify_interactions(rules):
    """
    Identify interaction rules involving multiple features and extract features involved.
    """
    logger.info("Identifying interactions in rules...")
    def extract_features_from_rule(rule):
        if isinstance(rule, str):
            terms = rule.split(' and ')
            features = set()
            for term in terms:
                # Terms look like 'feature_1 > 0.5' or 'feature_2 <= 1.0'
                feature = term.strip().split()[0]
                features.add(feature)
            return features
        else:
            return set()
    rules['features'] = rules['rule'].apply(extract_features_from_rule)
    rules['n_features'] = rules['features'].apply(len)
    interaction_rules = rules[rules['n_features'] > 1]
    logger.info(f"Identified {len(interaction_rules)} interaction rules.")
    return interaction_rules

def summarize_interactions(interaction_rules):
    """
    Summarize the feature interactions found in the interaction rules.
    """
    logger.info("Summarizing feature interactions...")
    # Create a list of sorted tuples of features for each rule
    interaction_rules['feature_combination'] = interaction_rules['features'].apply(lambda x: tuple(sorted(x)))
    # Group by feature combinations and sum the importance
    interaction_summary = interaction_rules.groupby('feature_combination').agg(
        importance_sum=('importance', 'sum'),
        importance_mean=('importance', 'mean'),
        count=('importance', 'count')
    ).reset_index()
    # Sort by importance_sum descending
    interaction_summary = interaction_summary.sort_values('importance_sum', ascending=False)
    return interaction_summary

def save_rules(rules, save_dir):
    """
    Save the extracted rules to a CSV file.
    """
    try:
        rules_path = os.path.join(save_dir, 'rules.csv')
        rules.to_csv(rules_path, index=False)
        logger.info(f"Rules saved to {rules_path}")
    except Exception as e:
        logger.error(f"Failed to save rules: {e}")
        raise

def save_interactions(interactions, save_dir):
    """
    Save the identified interactions to a CSV file.
    """
    try:
        interactions_path = os.path.join(save_dir, 'interactions.csv')
        interactions.to_csv(interactions_path, index=False)
        logger.info(f"Interactions saved to {interactions_path}")
    except Exception as e:
        logger.error(f"Failed to save interactions: {e}")
        raise

def save_interaction_summary(interaction_summary, save_dir):
    """
    Save the summarized feature interactions to a CSV file.
    """
    try:
        summary_path = os.path.join(save_dir, 'interaction_summary.csv')
        # Convert 'feature_combination' from tuple to string for saving
        interaction_summary['features'] = interaction_summary['feature_combination'].apply(lambda x: ', '.join(x))
        interaction_summary = interaction_summary.drop(columns=['feature_combination'])
        interaction_summary.to_csv(summary_path, index=False)
        logger.info(f"Interaction summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save interaction summary: {e}")
        raise

def process_model(model_name, model, X_train, save_dir):
    """
    Process each model to generate and save RuleFit explanations.
    """
    logger.info(f"Processing model: {model_name.upper()}")
    y_preds = model.predict(X_train)
    rulefit_model = fit_rulefit_model(X_train, y_preds)
    rules = extract_rules(rulefit_model)
    interactions = identify_interactions(rules)
    interaction_summary = summarize_interactions(interactions)
    save_rules(rules, save_dir)
    save_interactions(interactions, save_dir)
    save_interaction_summary(interaction_summary, save_dir)
    logger.info(f"RuleFit explanations for model {model_name.upper()} saved successfully.")

def main():
    """
    Main function to orchestrate the generation of RuleFit explanations for all models.
    """
    try:
        # Create explanation directories
        directories = create_directories(EXPLANATIONS_DIR, MODEL_NAMES)
        # Load data
        X_train, _ = load_data(DATA_PATH, TARGET_PATH)
        # Load models
        models = load_models(MODEL_PATHS)
        # Process each model
        for model_name, model in models.items():
            save_dir = directories[model_name]
            process_model(model_name, model, X_train, save_dir)
        logger.info("All RuleFit explanations generated and saved successfully.")
    except Exception as e:
        logger.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()