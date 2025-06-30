import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="CKD Stage Predictor", layout="centered")
st.title("🧬 Chronic Kidney Disease (CKD) Stage Predictor")
st.markdown("📋 Enter patient details or upload a CSV file to detect CKD and determine its stage.")

# Sidebar
st.sidebar.image("https://tse2.mm.bing.net/th?id=OIP.dc0PQ6gmNWvTwc7cxKrFbgHaHa", width=120)
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app predicts **CKD presence** using a trained ML model and determines the **CKD stage** using the **eGFR value**.  
It also provides **stage-specific treatment** and **lifestyle recommendations**.
""")

# --- Load model and preprocessing objects ---
def load_object(path):
    return joblib.load(path)

model_path = os.path.join("models", "ckd_best_model.joblib")
scaler_path = os.path.join("models", "scaler.joblib")
selector_path = os.path.join("models", "selector.joblib")

model = load_object(model_path)
scaler = load_object(scaler_path) if os.path.exists(scaler_path) else None
selector = load_object(selector_path) if os.path.exists(selector_path) else None

# --- CKD Staging from eGFR ---
def determine_ckd_stage(egfr):
    try:
        egfr = float(egfr)
        if egfr >= 90:
            return "Stage 1"
        elif egfr >= 60:
            return "Stage 2"
        elif egfr >= 30:
            return "Stage 3"
        elif egfr >= 15:
            return "Stage 4"
        else:
            return "Stage 5"
    except:
        return "Unknown"

stage_treatment = {
    "Stage 1": {
        "medications": (
            "- Control blood pressure (aim <130/80 mmHg) with ACE inhibitors or ARBs if indicated\n"
            "- Manage blood glucose tightly in diabetics\n"
            "- Avoid NSAIDs and other nephrotoxic drugs\n"
            "- Use lipid-lowering agents if needed"
        ),
        "lifestyle": (
            "- Maintain regular moderate exercise (e.g., 30 mins daily)\n"
            "- Follow a low-sodium diet (<2,300 mg/day)\n"
            "- Maintain healthy weight (BMI 18.5-24.9)\n"
            "- Avoid smoking and excessive alcohol\n"
            "- Stay hydrated but avoid overhydration"
        )
    },
    "Stage 2": {
        "medications": (
            "- Continue ACE inhibitors or ARBs for kidney protection\n"
            "- Start statins if hyperlipidemia present\n"
            "- Monitor and treat anemia if develops\n"
            "- Control blood pressure and glucose meticulously"
        ),
        "lifestyle": (
            "- Adopt a low protein diet (0.6-0.8 g/kg/day) if advised\n"
            "- Limit salt intake and avoid processed foods\n"
            "- Smoking cessation\n"
            "- Regular physical activity tailored to patient’s ability\n"
            "- Reduce stress and ensure good sleep hygiene"
        )
    },
    "Stage 3": {
        "medications": (
            "- Treat anemia with erythropoiesis-stimulating agents if indicated\n"
            "- Manage bone-mineral disorders (phosphate binders, vitamin D supplements)\n"
            "- Statins for cardiovascular risk reduction\n"
            "- Manage fluid overload with diuretics if needed"
        ),
        "lifestyle": (
            "- Limit potassium and phosphorus intake as per dietitian advice\n"
            "- Consult dietitian for tailored nutrition plan\n"
            "- Maintain physical activity but avoid overexertion\n"
            "- Avoid nephrotoxic drugs\n"
            "- Monitor and control body weight"
        )
    },
    "Stage 4": {
        "medications": (
            "- Prepare for renal replacement therapy (dialysis or transplant)\n"
            "- Treat complications like acidosis, hyperkalemia\n"
            "- Manage anemia and mineral bone disease aggressively\n"
            "- Optimize cardiovascular risk management"
        ),
        "lifestyle": (
            "- Frequent nephrology visits and monitoring\n"
            "- Patient education on dialysis options and lifestyle\n"
            "- Follow fluid restrictions and low-sodium diet\n"
            "- Avoid infection risks\n"
            "- Psychological support and counseling"
        )
    },
    "Stage 5": {
        "medications": (
            "- Initiate dialysis (hemodialysis or peritoneal dialysis) or prepare for kidney transplant\n"
            "- Use erythropoiesis-stimulating agents and iron supplements\n"
            "- Manage secondary hyperparathyroidism and other metabolic complications\n"
            "- Pain and symptom management"
        ),
        "lifestyle": (
            "- Strict dialysis diet (low potassium, low phosphorus, controlled protein)\n"
            "- Strict fluid restriction\n"
            "- Maintain dialysis schedule and access care\n"
            "- Prepare psychologically for transplant if eligible\n"
            "- Engage in light physical activity as tolerated"
        )
    }
}


# --- Categorical encoding ---
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

# --- Feature list ---
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
 # ← Copy your full feature_info list here

# --- Manual Input Form ---
st.header("📝 Input Patient Data Manually")
user_input = {}
col1, col2 = st.columns(2)
for idx, (feature, ftype) in enumerate(feature_info):
    with col1 if idx % 2 == 0 else col2:
        if isinstance(ftype, list):
            user_input[feature] = st.selectbox(f"{feature}", ftype)
        else:
            user_input[feature] = st.text_input(f"{feature}")

if st.button("🔍 Predict from Input"):
    try:
        input_data = {}
        for feature, ftype in feature_info:
            val = user_input[feature]
            if feature in categorical_maps:
                input_data[feature] = categorical_maps[feature][val]
            else:
                input_data[feature] = pd.to_numeric(val, errors='coerce')
        input_df = pd.DataFrame([input_data])
        st.write("🔧 Processed Input:")
        st.dataframe(input_df)

        if input_df.isnull().any().any():
            st.error("❌ Please fill all fields correctly.")
        else:
            if scaler:
                input_df = scaler.transform(input_df)
            if selector:
                input_df = selector.transform(input_df)
            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("✅ CKD Detected")
                egfr_value = user_input.get("Estimated Glomerular Filtration Rate (eGFR)", None)
                ckd_stage = determine_ckd_stage(egfr_value)
                st.markdown(f"### 🔬 CKD Stage: **{ckd_stage}**")
                if ckd_stage in stage_treatment:
                    st.subheader(f"🩺 Recommendations for {ckd_stage}")
                    st.markdown(f"**💊 Medications:**\n{stage_treatment[ckd_stage]['medications']}")
                    st.markdown(f"**🍎 Lifestyle:**\n{stage_treatment[ckd_stage]['lifestyle']}")
            else:
                st.success("🟢 No CKD Detected")
    except Exception as e:
        st.error(f"Error: {e}")

# --- CSV Upload ---
st.header("📁 Or Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in df_uploaded.columns:
            df_uploaded = df_uploaded.drop(columns=['Unnamed: 0'])
        st.write("📄 Uploaded Data:")
        st.dataframe(df_uploaded)

        input_data = df_uploaded.iloc[0].to_dict()
        for feature in feature_info:
            feat_name = feature[0]
            if feat_name in categorical_maps and feat_name in input_data:
                input_data[feat_name] = categorical_maps[feat_name].get(str(input_data[feat_name]).strip().lower(), 0)

        input_df = pd.DataFrame([input_data])
        st.write("🔧 Processed Input:")
        st.dataframe(input_df)

        if input_df.isnull().any().any():
            st.error("❌ Invalid/missing values in CSV.")
        else:
            if scaler:
                input_df = scaler.transform(input_df)
            if selector:
                input_df = selector.transform(input_df)
            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("✅ CKD Detected")
                egfr_value = input_data.get("Estimated Glomerular Filtration Rate (eGFR)", None)
                ckd_stage = determine_ckd_stage(egfr_value)
                st.markdown(f"### 🔬 CKD Stage: **{ckd_stage}**")
                if ckd_stage in stage_treatment:
                    st.subheader(f"🩺 Recommendations for {ckd_stage}")
                    st.markdown(f"**💊 Medications:**\n{stage_treatment[ckd_stage]['medications']}")
                    st.markdown(f"**🍎 Lifestyle:**\n{stage_treatment[ckd_stage]['lifestyle']}")
            else:
                st.success("🟢 No CKD Detected")
    except Exception as e:
        st.error(f"Error: {e}")
