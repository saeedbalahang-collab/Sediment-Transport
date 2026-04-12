# Transferability Limits of Machine Learning in Fluvial Sediment Transport

This repository contains code and resources for reproducing the results of our study on the transferability limits of machine learning models under climate-driven regime shifts in fluvial sediment transport. We trained an XGBoost model using hyperparameter tuning via Optuna and evaluated its performance on test data.

## Repository Contents

- Model_training.py: This script performs data normalization, model training, hyperparameter tuning, and model evaluation. Users can reproduce the results and obtain a new XGBoost model by running this script in Google Colab.
- Predictor.py: This script implements the trained XGBoost model (XGBoost.pkl) and scaler (scaler.pkl) on test data to reproduce predictions.
- data.csv: The full dataset used for training and evaluation the model.
- Test.csv: The test dataset used as sample data for trained XGBoost model.
- XGBoost.pkl: The trained XGBoost model.
- scaler.pkl: The scaler used for normalizing input features.
- requirements.txt: A list of required Python packages for running the scripts.

## Getting Started

To reproduce our results or implement predictions using our trained model, follow these steps:

1. Run the model training script in Google Colab:
   - Upload Model_training.py.
   - Execute the script to train and evaluate the model.
  
2. To use our trained XGBoost model and predict sample data (Test.csv is diffult file but you can change it to temporal_data.csv or spatial_data.csv file) in Google Colab:
   - Upload Predictor.py.
   - Execute the script.
  
4. To make predictions on test data, run Predictor.py after training.

## Code Availability

The code used in this study is available at: [https://github.com/saeedbalahang-collab/Sediment-Transport](https://github.com/saeedbalahang-collab/Sediment-Transport).

The trained models and scaler files can be found in this repository, allowing users to replicate our findings or apply the model to their own datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
   
   
