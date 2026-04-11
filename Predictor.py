!apt-get update -qq
!apt-get install git-lfs -qq
!git lfs install

!rm -rf Sediment-Transport
!git clone https://github.com/saeedbalahang-collab/Sediment-Transport.git

import os
os.chdir('Sediment-Transport')

import pandas as pd
import pickle
import joblib

print("Files in repo:", os.listdir())


try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded with joblib")
except Exception as e1:
    print("joblib failed:", e1)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded with pickle")

try:
    model = joblib.load('XGBoost.pkl')
    print("Model loaded with joblib")
except Exception as e2:
    print("joblib failed:", e2)
    with open('XGBoost.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded with pickle")

test_data = pd.read_csv('Test.csv')

predictor_columns = ['d*', 'D/d50', 'sheilds parameter', 'Frd', 'W/D']
print("Columns in Test.csv:", list(test_data.columns))

X_test = test_data[predictor_columns]
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

output_df = pd.concat(
    [test_data.reset_index(drop=True),
     pd.DataFrame(predictions, columns=['Predictions'])],
    axis=1
)

output_df.to_csv('/content/predictions_output.csv', index=False)
print(output_df)
