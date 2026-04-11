# Step 1: Clone the GitHub repository
!git clone https://github.com/saeedbalahang-collab/Sediment-Transport.git

# Step 2: Change directory to the cloned repository (if necessary)
import os
os.chdir('Sediment-Transport')

# Step 3: Import necessary libraries
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Step 4: Load the scaler and model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('XGBoost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Step 5: Load the test data
test_data = pd.read_csv('test.csv')

# Step 6: Select the relevant columns
predictor_columns = ['d*', 'D/d50', 'sheilds parameter', 'Frd', 'W/D']
X_test = test_data[predictor_columns]

# Step 7: Scale the features
X_test_scaled = scaler.transform(X_test)

# Step 8: Make predictions
predictions = model.predict(X_test_scaled)

# Step 9: Create a DataFrame with the predictions
output_df = pd.DataFrame(predictions, columns=['Predictions'])

# Optionally, concatenate the predictions with the original test data if needed
output_df = pd.concat([test_data.reset_index(drop=True), output_df], axis=1)

# Step 10: Save the output to a CSV file
output_df.to_csv('predictions_output.csv', index=False)

# Step 11: Display the DataFrame
print(output_df)
