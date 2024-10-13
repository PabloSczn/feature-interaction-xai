# Feature Interactions - XAI Project

## Project Overview 

This project aims to compare different Explainable AI (XAI) methods to assess how well they capture and explain **feature interactions** in machine learning models. The goal is to better understand the influence of individual features and their interactions in model decision-making.
The project implements the **Accumulated Local Effects (ALE)** method, with plans to include other methods such as **RuleFit** and **H-statistic** in the future.

### 🚧 Please note this project is on progress 🚧

Currently, the project includes:
- **Friedman #1** dataset generation for non-linear feature interaction simulation.
- Training of two machine learning models: **XGBoost** and **Random Forest**.
- Generation of **ALE plots** for both individual feature effects (1D ALE) and feature interactions (2D ALE).

## Setup and Installation

1. **Clone repository**:
```bash
   git clone https://github.com/PabloSczn/feature-interaction-xai.git
   cd feature-interaction-xai
```

2. **Create a Python virtual environment**:
```bash
   python -m venv venv
   venv\Scripts\activate  #On Unix systems: source venv/bin/activate
```

3. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

## Project Workflow

1. **Generate the dataset**:
   Generate the **Friedman #1** dataset and save it in the data/ directory:
```bash
   python generate-friedman-1.py
```

2. **Train models**:
   Train the **XGBoost** and **Random Forest** models and save them in the models/ directory:
```bash
   python train-models.py
```

3. **Generate ALE explanations**:
   Generate **ALE plots** for individual features and feature interactions:
```bash
   python methods/ale.py
```

The generated explanations will be saved for later comparison.

## Author
**Pablo Sanchez Narro**  
Contact: sancheznarro.pablo@gmail.com