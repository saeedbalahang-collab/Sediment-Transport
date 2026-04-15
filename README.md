# **Transferability Limits of Machine Learning in Fluvial Sediment Transport**

This repository provides the code, datasets, and trained models used to investigate the **transferability limits of machine learning models under climate-driven regime shifts in fluvial sediment transport**.

An **XGBoost-based model** was developed and optimized using Optuna-based hyperparameter tuning. Model performance is evaluated across multiple transfer scenarios (**test, spatial, and temporal datasets**), and **model interpretability is assessed using SHAP (SHapley Additive exPlanations)**.

---

## **Repository Structure**

* **`Model_training.py`**
  Data preprocessing, normalization, hyperparameter tuning, model training, and evaluation.

* **`Predictor.py`**
  Generates predictions using the trained model (`XGBoost.pkl`) and scaler (`scaler.pkl`).

* **`shap_analysis.py`**
  Performs SHAP-based interpretability analysis and generates feature importance visualizations for different datasets.

* **Datasets**

  * `data.csv` → Training dataset
  * `Test.csv` → Test dataset
  * `spatial_data.csv` → Spatial transfer dataset
  * `temporal_data.csv` → Temporal transfer dataset

* **Model Files**

  * `XGBoost.pkl` → Trained model
  * `scaler.pkl` → Feature scaler

* **Other**

  * `requirements.txt` → Required Python packages

---

## **Getting Started**

### **1. Run in Google Colab**

1. Open Google Colab
2. Create a new notebook
3. Install dependencies:

```python
!pip install -r requirements.txt
```

---

### **2. Upload Required Files**

Upload the following files:

* `Model_training.py`
* `Predictor.py`
* `shap_analysis.py`
* Any required datasets

---

## **Reproducing Model Training**

```python
!python Model_training.py
```

This will:

* Preprocess and normalize the dataset
* Perform hyperparameter optimization
* Train the XGBoost model
* Evaluate performance
* Save model outputs

---

## **Running Predictions**

```python
!python Predictor.py
```

### **Input Options**

* Default: `Test.csv`
* Alternatives:

  * `spatial_data.csv`
  * `temporal_data.csv`

⚠️ Ensure all datasets have the same feature structure as the training data.

---

## **Model Interpretability (SHAP Analysis)**

Model interpretability is implemented using **SHAP (SHapley Additive exPlanations)** to quantify the contribution of each feature to model predictions.

### **Run SHAP Analysis**

```python
!python shap_analysis.py
```

### **What the Script Does**

The `shap_analysis.py` script:

* Loads the trained XGBoost model and scaler
* Reads input datasets
* Applies feature scaling
* Computes SHAP values
* Generates summary plots for each dataset

### **Datasets Used**

The analysis is performed on:

* `Test.csv` (baseline evaluation)
* `spatial_data.csv` (spatial transferability)
* `temporal_data.csv` (temporal transferability)

### **Outputs**

The script generates three SHAP summary plots:

* `shap_test.png`
* `shap_spatial.png`
* `shap_temporal.png`

These plots illustrate how feature importance varies across different environmental and transfer scenarios, providing insight into **model robustness and transferability limits**.

---

## **Study Scope**

This study evaluates how machine learning models trained under specific hydrological conditions perform when applied to:

* **Unseen spatial domains**
* **Different temporal regimes**

The results highlight limitations in model generalization under **non-stat9ionary environmental conditions**.

---

## **Code Availability**

Repository:
[https://github.com/saeedbalahang-collab/Sediment-Transport](https://github.com/saeedbalahang-collab/Sediment-Transport)

This repository enables users to:

* Reproduce model training
* Generate predictions
* Perform SHAP-based interpretability analysis

---

## **License**

This project is licensed under the **MIT License**.

