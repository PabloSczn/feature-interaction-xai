import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Load the datasets
X_train = pd.read_csv('data/friedman_X_train.csv')
X_test = pd.read_csv('data/friedman_X_test.csv')
y_train = pd.read_csv('data/friedman_y_train.csv').values.ravel()  # Flatten to a 1D array
y_test = pd.read_csv('data/friedman_y_test.csv').values.ravel()  # Flatten to a 1D array

# Initialise models
xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)

# Train XGBoost model
xgb_model.fit(X_train, y_train)
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Evaluate XGBoost model
xgb_train_rmse = root_mean_squared_error(y_train, xgb_train_pred)
xgb_test_rmse = root_mean_squared_error(y_test, xgb_test_pred)
xgb_train_r2 = r2_score(y_train, xgb_train_pred)
xgb_test_r2 = r2_score(y_test, xgb_test_pred)

print(f"XGBoost - Train R²: {xgb_train_r2}, Test R²: {xgb_test_r2}")

# Train Random Forest model
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
rf_train_rmse = root_mean_squared_error(y_train, rf_train_pred)
rf_test_rmse = root_mean_squared_error(y_test, rf_test_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

print(f"Random Forest - Train R²: {rf_train_r2}, Test R²: {rf_test_r2}")

# Create models directory if it doesn't exist
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the models for later use
joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))

print("Models saved successfully in 'models/' directory.")