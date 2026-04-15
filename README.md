# **Transferability Limits of Machine Learning in Fluvial Sediment Transport**

This repository provides the code, data, and trained models used to investigate the **transferability limits of machine learning approaches under climate-driven regime shifts in fluvial sediment transport**.

An **XGBoost-based model** was developed and optimized using **Optuna-driven hyperparameter tuning**, and its predictive performance was evaluated across different transfer scenarios (test, spatial, and temporal datasets).

---

## **Repository Structure**

* **`Model_training.py`**
  Script for data preprocessing, normalization, hyperparameter tuning (Optuna), model training, and performance evaluation.

* **`Predictor.py`**
  Script for applying the trained model (`XGBoost.pkl`) and scaler (`scaler.pkl`) to new datasets.

* **`data.csv`**
  Complete dataset used for model training and validation.

* **`Test.csv`**
  Sample test dataset for evaluating model predictions.

* **`spatial_data.csv`**
  Dataset used for assessing spatial transferability.

* **`temporal_data.csv`**
  Dataset used for assessing temporal transferability.

* **`XGBoost.pkl`**
  Pre-trained XGBoost model.

* **`scaler.pkl`**
  Feature scaler used for input normalization.

* **`requirements.txt`**
  List of required Python packages.

---

## **Getting Started**

### **1. Run in Google Colab**

1. Open Google Colab
2. Create a new notebook
3. Install required dependencies:

```python
!pip install -r requirements.txt
```

---

### **2. Upload Required Files**

Upload the following files to your Colab environment:

* `Model_training.py`
* `Predictor.py`
* Dataset files (if needed)

---

## **Reproducing Model Training**

To reproduce the training process and results:

```python
!python Model_training.py
```

This script will:

* Preprocess and normalize the dataset
* Perform hyperparameter optimization using Optuna
* Train the XGBoost model
* Evaluate model performance
* Save the trained model and associated outputs

---

## **Running Predictions**

To generate predictions using the provided trained model:

```python
!python Predictor.py
```

### **Input Data**

* Default input: `Test.csv`
* Alternative datasets:

  * `spatial_data.csv` (spatial transferability)
  * `temporal_data.csv` (temporal transferability)

⚠️ **Important:**
All input datasets must have the same feature structure as the training data.

---

## **Output Files**

After execution, the following outputs will be generated:

* `XGBoost.pkl` → trained model
* `scaler.pkl` → feature scaler
* `metrics_summary.csv` → evaluation metrics
* Prediction outputs (from `Predictor.py`)

---

## **Study Scope**

This work focuses on evaluating how machine learning models trained under specific hydrological conditions perform when applied to:

* **Unseen spatial domains**
* **Different temporal regimes**

The results highlight important limitations in model generalization under **non-stationary environmental conditions**.

---

## **Code Availability**

The complete codebase is publicly available at:
👉 [https://github.com/saeedbalahang-collab/Sediment-Transport](https://github.com/saeedbalahang-collab/Sediment-Transport)

The repository includes all necessary resources to:

* Reproduce the study
* Validate results
* Apply the trained model to new datasets

---

## **License**

This project is licensed under the **MIT License**.
