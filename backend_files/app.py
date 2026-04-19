import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime # Import datetime for Store_Current_Age calculation
from huggingface_hub import hf_hub_download
import os

# Initialize the Flask application
sales_predictor_api = Flask("Sales Predictor")

# Define paths for the model and preprocessor relative to the app's directory
# Assuming joblib files are in the same directory as app.py within the container

MODEL_FILENAME = "sales_prediction_model_v1_0.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

# Load the model and preprocessor
try:
    model_path = hf_hub_download(
        repo_id="Rajse/Superkart-model",
        filename=MODEL_FILENAME
    )
    model = joblib.load(model_path)
    preprocessor_path = hf_hub_download(
        repo_id="Rajse/Superkart-model",
        filename=PREPROCESSOR_FILENAME
    )
    preprocessor = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

@sales_predictor_api.get('/')
def home():
    return "Welcome to the Superkart Sales Prediction API!"

@sales_predictor_api.post('/v1/sales')
def predict_sales():
    if not model or not preprocessor:
        return jsonify({'error': 'Model or preprocessor not loaded on server.'}), 500

    try:
        property_data = request.get_json(force=True)

        # Create a DataFrame from the raw input data (ensure all keys are present or handle missing)
        input_df_raw = pd.DataFrame([property_data])

        # --- Feature Engineering (replicate notebook steps exactly) ---
        # 1. Calculate Store_Current_Age
        current_year = datetime.now().year
        input_df_raw['Store_Current_Age'] = current_year - input_df_raw['Store_Establishment_Year']

        # 2. Transform Product_Type into 'Perishables' or 'Non Perishables'
        perishables = [
            "Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"
        ]
        input_df_raw['Product_Type'] = input_df_raw['Product_Type'].apply(
            lambda x: 'Perishables' if x in perishables else 'Non Perishables'
        )

        # 3. Drop 'Product_Id' and 'Store_Establishment_Year' (as these were dropped before model training)
        columns_to_drop = []
        if 'Product_Id' in input_df_raw.columns:
            columns_to_drop.append('Product_Id')
        if 'Store_Establishment_Year' in input_df_raw.columns:
            columns_to_drop.append('Store_Establishment_Year')

        # Drop columns, ignoring if they don't exist in the current input_df_raw
        input_df_engineered = input_df_raw.drop(columns=columns_to_drop, errors='ignore')

        # Ensure numerical features are correctly typed (convert from potential strings in JSON)
        numerical_cols_to_convert = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Current_Age']
        for col in numerical_cols_to_convert:
            if col in input_df_engineered.columns:
                input_df_engineered[col] = pd.to_numeric(input_df_engineered[col], errors='coerce')
                # If conversion results in NaN, it means an invalid numeric value was passed
                if input_df_engineered[col].isnull().any():
                    return jsonify({'error': f"Invalid numeric value provided for '{col}'."}), 400

        # 4. Transform the engineered data using the preprocessor
        processed_data = preprocessor.transform(input_df_engineered)

        # 5. Make prediction (no np.exp as target was not log-transformed)
        predicted_sales = model.predict(processed_data)[0]

        predicted_price = round(float(predicted_sales), 2)
        return jsonify({'Predicted Price (in dollars)': predicted_price})

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 400


@sales_predictor_api.post('/v1/salebatch')
def predict_sales_batch():
    # ... (loading checks) ...
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        df = pd.read_csv(file)

        # --- OPTIMIZED FEATURE ENGINEERING ---
        # Vectorized year calculation (much faster than .apply)
        df['Store_Current_Age'] = datetime.now().year - df['Store_Establishment_Year']

        # Vectorized product type (using .isin is significantly faster)
        perishables = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]
        df['Product_Type'] = np.where(df['Product_Type'].isin(perishables), 'Perishables', 'Non Perishables')

        # Drop columns efficiently
        to_drop = [c for c in ['Product_Id', 'Store_Establishment_Year'] if c in df.columns]
        df_engineered = df.drop(columns=to_drop)

        # Vectorized numeric conversion
        num_cols = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Current_Age']
        df_engineered[num_cols] = df_engineered[num_cols].apply(pd.to_numeric, errors='coerce')

        # --- PREDICTION ---
        processed = preprocessor.transform(df_engineered)
        preds = model.predict(processed)

        # Build response dictionary
        p_ids = df['Product_Id'] if 'Product_Id' in df.columns else range(len(preds))
        output = {str(pid): round(float(p), 2) for pid, p in zip(p_ids, preds)}

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    sales_predictor_api.run(host='0.0.0.0', port=7860, debug=True)
