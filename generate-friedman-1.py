import os
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split


'''
Friedman #1 Dataset: A synthetic dataset that has non-linear interactions between features.
  - In this dataset, the target is generated as a function of multiple features, some of which interact in complex ways:
        y(X) = 10 * sin(pi * X_1 * X_2) + 20 * (X_3 - 0.5) ** 2 + 10 * X_4 + 5 * X_5 + noise * N(0, 1).

  - This function contains a clear interaction between `x_1` and `x_2`, which can be useful for benchmarking how well each method captures this known interaction.

    Link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html  
'''

# Create data directory if it doesn't exist
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Generate Friedman #1 Dataset
X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to pandas DataFrames for easier saving
X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
y_train_df = pd.DataFrame(y_train, columns=["target"])
y_test_df = pd.DataFrame(y_test, columns=["target"])

# Save datasets to the data directory
X_train_df.to_csv(os.path.join(data_dir, "friedman_X_train.csv"), index=False)
X_test_df.to_csv(os.path.join(data_dir, "friedman_X_test.csv"), index=False)
y_train_df.to_csv(os.path.join(data_dir, "friedman_y_train.csv"), index=False)
y_test_df.to_csv(os.path.join(data_dir, "friedman_y_test.csv"), index=False)

print(f"Datasets saved in '{data_dir}/' directory.")
