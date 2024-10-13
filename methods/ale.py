import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PyALE import ale

# Load the dataset
X_train = pd.read_csv('./data/friedman_X_train.csv')

# Load the pre-trained models from the models directory
xgb_model_path = './models/xgb_model.pkl'
rf_model_path = './models/rf_model.pkl'

# Check if the models exist
if os.path.exists(xgb_model_path) and os.path.exists(rf_model_path):
    xgb_model = joblib.load(xgb_model_path)
    rf_model = joblib.load(rf_model_path)
else:
    raise FileNotFoundError("Trained models not found in 'models/' directory. Please run 'train_models.py' to train and save the models.")

# Ensure the output directory for ALE plots exists
plot_dir = './explanations/ale/plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Function to generate ALE explanations for a given model and feature(s)
def generate_ale_explanations(model, model_name, X_train):
    """
    Generates ALE plots for both 1D and 2D feature interactions.

    Explanations:
    - 1D ALE plot: Shows the effect of each feature on the model’s predictions, independent of other features.
    - 2D ALE plot: Visualises interactions between two features and how their joint effects contribute to the model’s decision-making.

    Plots will be saved for comparison purposes later.
    """
    print(f"Generating ALE explanations for {model_name}...")

    # 1D ALE: Analyse individual feature effects
    for feature in X_train.columns:
        ale_eff = ale(X=X_train, model=model, feature=[feature], grid_size=50, include_CI=True)
        
        # Plot 1D ALE
        plt.figure()
        ale_eff.plot()
        plt.title(f"{model_name} - ALE for {feature}")
        plt.savefig(f'{plot_dir}{model_name}_ALE_{feature}.png')
        plt.close()

    # 2D ALE: Analyse interactions between feature pairs
    for i, feature_1 in enumerate(X_train.columns[:-1]):  # Loop through all features except the last one
        for feature_2 in X_train.columns[i+1:]:
            ale_eff = ale(X=X_train, model=model, feature=[feature_1, feature_2], grid_size=50)
            
            # Plot 2D ALE
            plt.figure()
            ale_eff.plot()
            plt.title(f"{model_name} - ALE Interaction: {feature_1} and {feature_2}")
            plt.savefig(f'{plot_dir}{model_name}_ALE_Interaction_{feature_1}_{feature_2}.png')
            plt.close()

# Generate ALE explanations for XGBoost and Random Forest models
generate_ale_explanations(xgb_model, 'XGBoost', X_train)
generate_ale_explanations(rf_model, 'RandomForest', X_train)

print("ALE explanations generated and saved in 'explanations/ale/plots/' directory.")
