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

Follow the steps below to reproduce the results or use the trained model for prediction.

1. Open in Google Colab
- Go to Google Colab
- Create a new notebook

2. Install Required Libraries in requirements.txt
3. Upload the following files to the Google Colab:
- Model_training.py
- Predictor.py

4. Train the Model (Reproduce Results)

- Run the training script:

!python Model_training.py

This will:

- Preprocess the dataset
- Perform hyperparameter tuning using Optuna
- Train the XGBoost model
- Evaluate model performance
- Save outputs (model, scaler, metrics)

5. Run Predictions Using the Trained Model

To generate predictions using  our provided model:

!python Predictor.py

📌 Notes:

The default input file is Test.csv
You can replace it with temporal_data.csv or spatial_data.csv
Ensure the input data has the same feature structure as the training data

6. Output Files
After execution, the following files will be generated:
- XGBoost.pkl → trained model
- scaler.pkl → feature scaler
- metrics_summary.csv → performance metrics
- prediction outputs (if using Predictor.py).

## Code Availability

The code used in this study is available at: [https://github.com/saeedbalahang-collab/Sediment-Transport](https://github.com/saeedbalahang-collab/Sediment-Transport).

The trained models and scaler files can be found in this repository, allowing users to replicate our findings or apply the model to their own datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
   
   
