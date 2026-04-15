
# Install necessary libraries
!pip install joblib shap matplotlib requests xgboost

# Import libraries
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import requests

# -----------------------------
# Download model and scaler
# -----------------------------
model_url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/XGBoost.pkl'
scaler_url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/scaler.pkl'

with open('XGBoost.pkl', 'wb') as f:
    f.write(requests.get(model_url).content)

with open('scaler.pkl', 'wb') as f:
    f.write(requests.get(scaler_url).content)

# Load model and scaler
model = joblib.load('XGBoost.pkl')
scaler = joblib.load('scaler.pkl')

# -----------------------------
# Function to process dataset
# -----------------------------
def process_dataset(url, output_name):
    print(f"\nProcessing: {output_name}")

    # Load data
    data = pd.read_csv(url)

    # Show columns (for debugging if needed)
    print("Columns:", data.columns.tolist())

    # Select features (adjust if needed)
    X = data[['d*', 'D/d50', 'sheilds parameter', 'Frd', 'W/D']]

    # Scale
    X_scaled = scaler.transform(X)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Plot
    shap.summary_plot(
        shap_values,
        X,
        feature_names=['d$^*$', 'D/$d_{50}$', r'$\theta$', 'Fr$_{d}$', 'W/D'],
        show=False
    )

    # Save
    filename = f"{output_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")

# -----------------------------
# Dataset URLs
# -----------------------------
test_url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/Test.csv'
spatial_url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/spatial_data.csv'
temporal_url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/temporal_data.csv'

# -----------------------------
# Run for all datasets
# -----------------------------
process_dataset(test_url, 'shap_test')
process_dataset(spatial_url, 'shap_spatial')
process_dataset(temporal_url, 'shap_temporal')

print("\nAll three SHAP plots generated successfully.")
