# Feature Interactions - XAI Project

## Project Overview

This project aims to compare different Explainable AI (XAI) methods to assess how well they capture and explain **feature interactions** in machine learning models. The goal is to better understand the influence of individual features and their interactions in model decision-making.  
The project implements the **Accumulated Local Effects (ALE)**, **H-statistic**, and **SHAP interaction** methods, with plans to include other methods in the future.

### 🚧 Please note this project is in progress 🚧

Currently, the project includes:
- **Friedman #1** dataset generation for non-linear feature interaction simulation, with plans to implement the integration of another dataset that reflects a real-life scenario.
- **Bike Sharing** dataset generation. The data was made openly available by Capital-Bikeshare. Fanaee-T and Gama (2013) added weather data and seasonal information. The aim is to predict how many bikes will be rented depending on the weather and day.
- Training of two machine learning models: **XGBoost** and **Random Forest**.
- Generation of explanations for the different methods.

## Setup and Installation

1. **Clone repository**:
```bash
   git clone https://github.com/PabloSczn/feature-interaction-xai.git
   cd feature-interaction-xai
```

2. **Create a Python virtual environment**:
```bash
   python -m venv venv
   venv\Scripts\activate  # On Unix systems: source venv/bin/activate
```

3. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

## Project Workflow

1. **Generate the dataset**:
   - Generate the **Friedman #1** dataset and save it in `data/`:
   ```bash
   python generate-data/generate-friedman-1.py
   ```

   - Process the **Bike Sharing** dataset and save it in `data/`:
   ```bash
   python generate-data/generate-bike-sharing-data.py
   ```
      *Data source for Bike Sharing: Fanaee-T, H. (2013). [Bike Sharing Dataset](https://doi.org/10.24432/C5W894). UCI Machine Learning Repository.*

2. **Train models**:
   Train the **XGBoost** and **Random Forest** models and save them in the `models/` directory:
   ```bash
   python train-models.py
   ```
   This saves the models trained for both Bike Sharing and Friedman 1 datasets.

3. **Generate explanations**:
   Generate the explanations for the different methods:
   ```bash
   python methods/ale.py
   python methods/h-statistic.py
   python methods/shap-interaction.py
   ```
   - To generate the explanations for both dataset, you need to change `DATASET_NAME` to either `friedman1` or `bike-sharing`
   - The generated explanations will be saved for later comparison in `explanations/`.

## Author
**Pablo Sanchez Narro**  
Contact: sancheznarro.pablo@gmail.com