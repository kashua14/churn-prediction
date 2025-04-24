import streamlit as st
import pandas as pd
import joblib
import os

# === Page Config ===
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# === Title ===
st.title("📉 Customer Churn Prediction")
st.markdown("Upload customer data to predict churn. The app will use the features from your trained model.")

# === Load model safely ===
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

MODEL_PATH = "model.pkl"
model = load_model(MODEL_PATH)

# === Sidebar: Auto-generate CSV template with dummy values ===
st.sidebar.header("📁 Download Model-Based CSV Template")

if model:
    try:
        feature_columns = model.feature_names_in_.tolist()

        # Create dummy sample data with realistic values
        sample_data = {col: None for col in feature_columns}

        # Optional: Add smart defaults for known features
        for col in feature_columns:
            if "age" in col.lower():
                sample_data[col] = 30
            elif "gender" in col.lower():
                sample_data[col] = "Male"
            elif "segment" in col.lower():
                sample_data[col] = 4
            elif "mb" in col.lower():
                sample_data[col] = 1000
            elif "revenue" in col.lower() or "money" in col.lower():
                sample_data[col] = 50000
            elif "calls" in col.lower() or "sms" in col.lower():
                sample_data[col] = 100
            elif "device" in col.lower() or "handset" in col.lower():
                sample_data[col] = 9
            elif "lat" in col.lower():
                sample_data[col] = 0.35
            elif "long" in col.lower():
                sample_data[col] = 32.58
            elif "rec" in col.lower() or "q_rec" in col.lower():
                sample_data[col] = 1
            elif "is_" in col.lower():
                sample_data[col] = 1
            elif "manufacturer" in col.lower():
                sample_data[col] = 0
            elif "model" in col.lower():
                sample_data[col] = 4
            elif "gender" in col.lower():
                sample_data[col] = 2
            else:
                sample_data[col] = 0  # fallback

        sample_template = pd.DataFrame([sample_data])

        st.sidebar.download_button(
            label="⬇️ Download CSV Template with Sample Data",
            data=sample_template.to_csv(index=False),
            file_name="churn_template.csv",
            mime="text/csv"
        )
    except Exception:
        st.sidebar.warning("⚠️ Unable to extract feature names from the model.")
else:
    st.sidebar.warning("⚠️ Model not found. Upload `model.pkl` to use this app.")
    feature_columns = []

# === File Upload ===
st.subheader("📤 Upload Customer CSV")
uploaded_file = st.file_uploader("Upload a CSV file matching the template", type=["csv"])

if not model:
    st.warning("⚠️ Model file not found. Please upload `model.pkl` to the app directory.")
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview")
        st.dataframe(df.head())

        # === Validate columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            st.error("❌ Your file is missing the following required columns:")
            st.code(missing_cols)
        else:
            # Predict
            X_new = df[feature_columns].copy()
            predictions = model.predict(X_new)

            # Handle prediction probabilities safely
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_new)
                probabilities = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                probabilities = ["N/A"] * len(X_new)

            # Append to results
            df['Churn_Predicted'] = predictions
            df['Churn_Probability'] = probabilities

            st.success("✅ Prediction Completed")
            st.dataframe(df[['Churn_Predicted', 'Churn_Probability']])

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="churn_predictions_results.csv",
                mime='text/csv'
            )
    except Exception as e:
        st.error("❌ Failed to process your file.")
        st.exception(e)
else:
    st.info("📄 Please upload a customer dataset to begin.")
