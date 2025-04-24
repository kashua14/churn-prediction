import streamlit as st
import pandas as pd
import joblib
import os

# === Page Config ===
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# === Title ===
st.title("📉 Customer Churn Prediction")
st.markdown("Upload customer data to predict churn. Ensure your file matches the expected format.")

# === Required Columns ===
required_features = [
    'Mobile_Money', 'Voice_PAYG', 'Latitude', 'SMS_Bundle_Rev', 'Calls', 'Mbs', 'Longitude',
    'Manufacturer', 'Model_Name', 'Device_Type', 'Handset_Type', 'Gender', 'Value_Segment',
    'Age_Groups', 'REC_30_Days', 'REC_90_Days', 'Q_Rec', 'Q_Rec_over100mbs', 'Recon',
    'Gross_Add', 'Win_Backs', 'IS_NEW_SUBSCRIBER', 'IS_RECENTLY_ACTIVE', 'IS_BASE_ACTIVE',
    'IS_ENGAGED_USER', 'IS_HEAVY_USER', 'IS_WINBACK', 'IS_RECONNECTED', 'IS_PREMIUM_DEVICE',
    'DEVICE_CATEGORY'
]

# === Load model safely ===
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

MODEL_PATH = "best_model_SMOTE_XGBoost.pkl"
model = load_model(MODEL_PATH)

# === Sidebar: Sample Template ===
st.sidebar.header("📁 Download Sample CSV Template")
sample_data = pd.DataFrame([{
    'Mobile_Money': 35000,
    'Voice_PAYG': 1200,
    'Latitude': 0.3476,
    'SMS_Bundle_Rev': 500,
    'Calls': 45,
    'Mbs': 1500,
    'Longitude': 32.5825,
    'Manufacturer': 'Samsung',
    'Model_Name': 'Galaxy S20',
    'Device_Type': 'Smartphone',
    'Handset_Type': '4G',
    'Gender': 'Male',
    'Value_Segment': 'High',
    'Age_Groups': '25-34',
    'REC_30_Days': 1,
    'REC_90_Days': 1,
    'Q_Rec': 1,
    'Q_Rec_over100mbs': 1,
    'Recon': 0,
    'Gross_Add': 0,
    'Win_Backs': 0,
    'IS_NEW_SUBSCRIBER': 0,
    'IS_RECENTLY_ACTIVE': 1,
    'IS_BASE_ACTIVE': 1,
    'IS_ENGAGED_USER': 1,
    'IS_HEAVY_USER': 1,
    'IS_WINBACK': 0,
    'IS_RECONNECTED': 0,
    'IS_PREMIUM_DEVICE': 1,
    'DEVICE_CATEGORY': 'smartphone_4g'
}])

st.sidebar.download_button(
    label="⬇️ Download CSV Template",
    data=sample_data.to_csv(index=False),
    file_name="churn_template.csv",
    mime="text/csv"
)

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

        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            st.error("❌ Missing required columns:")
            st.code(missing_cols)
        else:
            # Extract only required features
            X_new = df[required_features].copy()

            # Predict classes
            predictions = model.predict(X_new)

            # Safely predict probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_new)
                probabilities = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
            else:
                probabilities = ["N/A"] * len(X_new)

            # Append to DataFrame
            df['Churn_Predicted'] = predictions
            df['Churn_Probability'] = probabilities

            st.success("✅ Prediction Completed")
            st.dataframe(df[['Churn_Predicted', 'Churn_Probability']])

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
    st.info("📄 Please upload your customer dataset to get started.")
