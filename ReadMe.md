# Superkart Sales Prediction MLOps Pipeline

## Overview
This project builds an end-to-end MLOps pipeline for predicting product sales in a retail setting using machine learning. It covers model training, preprocessing, experiment tracking, model versioning, backend API serving, and deployment-ready integration.

The solution is designed around a sales prediction use case where product and store attributes are used to estimate expected sales. The workflow includes data preprocessing, XGBoost-based model training, MLflow experiment tracking, artifact storage, and backend inference deployment.

---

## Project Objectives
- Build a machine learning model to predict Superkart product sales
- Create a reusable preprocessing and training pipeline
- Track experiments and model metrics using MLflow
- Store trained artifacts for reproducibility and deployment
- Serve predictions through a backend API
- Prepare the solution for deployment on Hugging Face Spaces

---

## Features
- End-to-end regression pipeline for sales prediction
- Automated preprocessing for numerical and categorical features
- Hyperparameter tuning using GridSearchCV
- Experiment tracking with MLflow
- Model artifact versioning using joblib
- Hugging Face Hub integration for model storage
- Backend API for single and batch predictions
- Deployment-friendly structure for production usage

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- MLflow
- Hugging Face Hub
- Flask / FastAPI backend serving
- Streamlit frontend
- Joblib

---

## Workflow
1. Load training and testing datasets
2. Identify numerical and categorical features
3. Apply preprocessing using scaling and encoding
4. Train an XGBoost regressor inside a pipeline
5. Tune model hyperparameters using GridSearchCV
6. Evaluate performance using regression metrics
7. Log experiments and metrics with MLflow
8. Save the trained model artifact
9. Upload the model to Hugging Face Hub
10. Use the model in a backend API for predictions

---

## Model Input Features
Typical prediction inputs include:
- Product_Id
- Product_Weight
- Product_Sugar_Content
- Product_Allocated_Area
- Product_Type
- Product_MRP
- Store_Id
- Store_Establishment_Year
- Store_Size
- Store_Location_City_Type
- Store_Type

Some features are transformed during inference, such as:
- Store_Current_Age
- Product_Type grouping into perishables / non-perishables

---

## Training Pipeline
The training pipeline uses:
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `ColumnTransformer` for feature-level preprocessing
- `Pipeline` to combine preprocessing and model training
- `XGBRegressor` for regression modeling

This ensures preprocessing and model logic are bundled together consistently for training and inference.

---

## Evaluation Metrics
The project evaluates model performance using:
- RMSE
- MAE
- R-squared
- Adjusted R-squared
- MAPE

These metrics are logged into MLflow for experiment comparison and monitoring.

---

## MLflow Integration
MLflow is used for:
- experiment tracking
- logging hyperparameters
- storing evaluation metrics
- saving model artifacts
- comparing different model runs

Example experiment name:
`mlops-training-experiment`

---

## Hugging Face Integration
The trained model is uploaded to Hugging Face Hub for centralized artifact storage and deployment support.

### Example repositories
- Dataset repo: `Rajse/Superkart-Dataset`
- Model repo: `Rajse/Superkart-model`
- Backend Space: `Rajse/Superkart-SalesPredictionBackend`

---

## API Endpoints
The backend supports prediction endpoints such as:

### Single prediction
`POST /v1/sales`

### Batch prediction
`POST /v1/salebatch`

These endpoints accept raw feature inputs and return predicted sales values.

---

## Project Structure
```bash
.
├── model_building/
│   └── train.py
├── backend_files/
│   └── app.py
├── deployment/
│   ├── Dockerfile
│   └── requirements.txt
├── model/
│   └── sales_prediction_model_v1_0.joblib
├── frontend/
│   └── app.py
├── README.md
└── requirements.txt