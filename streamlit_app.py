import streamlit as st
import pandas as pd
import joblib
import os

# === Page Config ===
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# === Title ===
st.title("📉 Customer Churn Prediction")
st.markdown("Upload your customer data to predict churn likelihood using our machine learning model.")

# === Try loading the model safely ===
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

MODEL_PATH = "best_model.pkl"
model = load_model(MODEL_PATH)

# === Sample Template for Users ===
st.sidebar.header("📁 Download Sample CSV Template")
sample_data = pd.DataFrame({
    'AGE': [35],
    'MBS': [1200],
    'CALLS': [45],
    'MOBILE_MONEY': [30000],
    'VOICE_PAYG': [5000],
    'VALUESEGMENT_High': [1],
    'GENDER_MALE': [1]
})
st.sidebar.download_button(
    label="⬇️ Download CSV Template",
    data=sample_data.to_csv(index=False),
    file_name="churn_template.csv",
    mime="text/csv"
)

# === File Upload ===
st.subheader("📤 Upload Customer CSV File")
uploaded_file = st.file_uploader("Upload a file with the required columns", type=["csv"])

if not model:
    st.warning("⚠️ The model file was not found (`best_model.pkl`). Please upload it to the project directory.")
elif uploaded_file:
    # === Read uploaded CSV ===
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data Preview")
        st.dataframe(df.head())

        # Check that all required columns are present
        required_cols = model.feature_names_in_
        if all(col in df.columns for col in required_cols):
            X_new = df[required_cols]

            # Predict
            predictions = model.predict(X_new)
            probabilities = model.predict_proba(X_new)[:, 1]

            # Add predictions to output
            df['Churn_Predicted'] = predictions
            df['Churn_Probability'] = probabilities

            # Show results
            st.success("✅ Predictions Complete")
            st.dataframe(df[['Churn_Predicted', 'Churn_Probability']])

            # Allow CSV download
            st.download_button(
                label="📥 Download Results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="churn_predictions.csv",
                mime='text/csv'
            )
        else:
            st.error("❌ Your file is missing required columns:")
            missing = [col for col in required_cols if col not in df.columns]
            st.code(missing)
    except Exception as e:
        st.error("❌ Error reading the uploaded CSV file:")
        st.exception(e)
else:
    st.info("📄 Please upload a customer dataset in CSV format.")
