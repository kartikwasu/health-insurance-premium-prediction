from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "insurance_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Insurance Premium Predictor", page_icon="📊", layout="centered")
st.title("🏥 Health Insurance Premium Prediction")
st.write("Estimate insurance charges with a machine-learning model.")
st.info("Demo project: the included dataset is synthetic. Predictions are educational estimates, not financial or insurance advice.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Train it first using: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 28)
        sex = st.selectbox("Sex", ["female", "male"])
        bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=27.5, step=0.1)
    with col2:
        children = st.selectbox("Number of children", [0, 1, 2, 3, 4, 5])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submitted = st.form_submit_button("Predict premium")

if submitted:
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region
    }])
    prediction = float(model.predict(sample)[0])
    st.success(f"Estimated charges: ₹{prediction:,.2f}")
    st.caption("The model was trained on synthetic demo data; this number is not a real-world quote.")

if METRICS_PATH.exists():
    with st.expander("Model evaluation (demo dataset)"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(metrics)
