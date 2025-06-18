import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="CKD Predictor", layout="centered")
st.title("Chronic Kidney Disease Predictor")
st.markdown("""Enter patient data below and get instant predictions.
""")

# Load model and preprocessing objects
def load_object(path):
    return joblib.load(path)

model_path = os.path.join("models", "ckd_best_model.joblib")
scaler_path = os.path.join("models", "scaler.joblib")
selector_path = os.path.join("models", "selector.joblib")
encoder_path = os.path.join("models", "encoder.joblib")

model = load_object(model_path)
scaler = load_object(scaler_path) if os.path.exists(scaler_path) else None
selector = load_object(selector_path) if os.path.exists(selector_path) else None
encoder = load_object(encoder_path) if os.path.exists(encoder_path) else None

# All features from the dataset header (update categorical types as needed)
feature_info = [
    ("Age of the patient", "number"),
    ("Blood pressure (mm/Hg)", "number"),
    ("Specific gravity of urine", "number"),
    ("Albumin in urine", "number"),
    ("Sugar in urine", "number"),
    ("Red blood cells in urine", ["normal", "abnormal"]),
    ("Pus cells in urine", ["normal", "abnormal"]),
    ("Pus cell clumps in urine", ["present", "not present"]),
    ("Bacteria in urine", ["present", "not present"]),
    ("Random blood glucose level (mg/dl)", "number"),
    ("Blood urea (mg/dl)", "number"),
    ("Serum creatinine (mg/dl)", "number"),
    ("Sodium level (mEq/L)", "number"),
    ("Potassium level (mEq/L)", "number"),
    ("Hemoglobin level (gms)", "number"),
    ("Packed cell volume (%)", "number"),
    ("White blood cell count (cells/cumm)", "number"),
    ("Red blood cell count (millions/cumm)", "number"),
    ("Hypertension (yes/no)", ["yes", "no"]),
    ("Diabetes mellitus (yes/no)", ["yes", "no"]),
    ("Coronary artery disease (yes/no)", ["yes", "no"]),
    ("Appetite (good/poor)", ["good", "poor"]),
    ("Pedal edema (yes/no)", ["yes", "no"]),
    ("Anemia (yes/no)", ["yes", "no"]),
    ("Estimated Glomerular Filtration Rate (eGFR)", "number"),
    ("Urine protein-to-creatinine ratio", "number"),
    ("Urine output (ml/day)", "number"),
    ("Serum albumin level", "number"),
    ("Cholesterol level", "number"),
    ("Parathyroid hormone (PTH) level", "number"),
    ("Serum calcium level", "number"),
    ("Serum phosphate level", "number"),
    ("Family history of chronic kidney disease", ["yes", "no"]),
    ("Smoking status", ["yes", "no"]),
    ("Body Mass Index (BMI)", "number"),
    ("Physical activity level", ["low", "medium", "high"]),
    ("Duration of diabetes mellitus (years)", "number"),
    ("Duration of hypertension (years)", "number"),
    ("Cystatin C level", "number"),
    ("Urinary sediment microscopy results", ["normal", "abnormal"]),
    ("C-reactive protein (CRP) level", "number"),
    ("Interleukin-6 (IL-6) level", "number")
]

st.header("Input Patient Data")
user_input = {}
for feature, ftype in feature_info:
    if isinstance(ftype, list):
        user_input[feature] = st.selectbox(f"{feature}", ftype)
    else:
        user_input[feature] = st.text_input(f"{feature}")

# Manual encoding for categorical fields (must match label encoding in training)
categorical_maps = {
    "Red blood cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cell clumps in urine": {"not present": 0, "present": 1},
    "Bacteria in urine": {"not present": 0, "present": 1},
    "Hypertension (yes/no)": {"no": 0, "yes": 1},
    "Diabetes mellitus (yes/no)": {"no": 0, "yes": 1},
    "Coronary artery disease (yes/no)": {"no": 0, "yes": 1},
    "Appetite (good/poor)": {"good": 0, "poor": 1},
    "Pedal edema (yes/no)": {"no": 0, "yes": 1},
    "Anemia (yes/no)": {"no": 0, "yes": 1},
    "Family history of chronic kidney disease": {"no": 0, "yes": 1},
    "Smoking status": {"no": 0, "yes": 1},
    "Physical activity level": {"low": 0, "medium": 1, "high": 2},
    "Urinary sediment microscopy results": {"normal": 0, "abnormal": 1}
}

if st.button("Predict"):
    try:
        input_data = {}
        for feature, ftype in feature_info:
            val = user_input[feature]
            if feature in categorical_maps:
                input_data[feature] = categorical_maps[feature][val]
            else:
                input_data[feature] = pd.to_numeric(val, errors='coerce')
        input_df = pd.DataFrame([input_data])
        st.write("Processed input for model:")
        st.dataframe(input_df)
        if input_df.isnull().any().any():
            st.error("Please fill all fields with valid values.")
        else:
            if scaler:
                input_df = scaler.transform(input_df)
            if selector:
                input_df = selector.transform(input_df)
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD Detected'}")
    except Exception as e:
        st.error(f"Error: {e}")

# Add file upload for CSV input
st.header("Or Upload Patient Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file with patient data", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        # Remove index column if present
        if 'Unnamed: 0' in df_uploaded.columns:
            df_uploaded = df_uploaded.drop(columns=['Unnamed: 0'])
        st.write("Uploaded Data:")
        st.dataframe(df_uploaded)
        # Use the first row for prediction
        input_data = df_uploaded.iloc[0].to_dict()
        # Map categorical fields
        for feature, ftype in feature_info:
            if feature in categorical_maps and feature in input_data:
                val = input_data[feature]
                if not isinstance(val, (int, float)):
                    input_data[feature] = categorical_maps[feature].get(str(val).strip().lower(), 0)
        input_df = pd.DataFrame([input_data])
        st.write("Processed input for model:")
        st.dataframe(input_df)
        if input_df.isnull().any().any():
            st.error("Please fill all fields with valid values in your CSV.")
        else:
            if scaler:
                input_df = scaler.transform(input_df)
            if selector:
                input_df = selector.transform(input_df)
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD Detected'}")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
