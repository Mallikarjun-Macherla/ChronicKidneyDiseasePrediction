# Chronic Kidney Disease Prediction

This project predicts the likelihood of Chronic Kidney Disease (CKD) using machine learning. It includes data preprocessing, model training, evaluation, and a user-friendly webapp for real-time predictions.

## Features
- Data cleaning, encoding, and feature selection
- Model training and hyperparameter tuning
- Streamlit webapp for easy predictions
- Supports both manual input and CSV upload

## Project Structure
```
├── ChronicKidneyDiseasePrediction.ipynb                # Advanced modeling and feature engineering
├── ckd_webapp.py                                       # Streamlit webapp for predictions
├── requirements.txt                                    # Python dependencies
├── models/                                             # Saved model and preprocessing objects
│   ├── ckd_best_model.joblib
│   ├── scaler.joblib
│   ├── selector.joblib
│   └── encoder.joblib
├── kidney_disease_dataset.csv                          # Main dataset
├── test_ckd_positive.csv                               # Example CKD-positive test case
├── test_ckd_negative.csv                               # Example CKD-negative test case
```

## How to Run
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Train the model:**
   - Run the Jupyter notebooks to preprocess data and train the model.
   - Preprocessing objects and the model will be saved in the `models/` folder.
3. **Start the webapp:**
   ```
   streamlit run ckd_webapp.py
   ```
   or (if `streamlit` is not recognized):
   ```
   python -m streamlit run ckd_webapp.py
   ```
4. **Use the webapp:**
   - Enter patient data manually or upload a CSV file for prediction.

## Notes
- Ensure the `models/` folder contains all required `.joblib` files.
- Feature names and order must match between training and prediction.
- Example test files are provided for quick validation.
