# Health Insurance Premium Prediction

A beginner-friendly end-to-end Data Science / Machine Learning project. It demonstrates loading data, exploratory analysis, preprocessing, model training, evaluation, saving a model, and serving predictions through Streamlit.

> **Important:** `data/insurance.csv` is a **synthetic demonstration dataset generated for this repository**. It is not real customer data. Evaluation metrics and predictions are for learning/demo purposes only and must not be presented as validated insurance estimates.

## Features
- Exploratory data analysis (EDA)
- Numeric scaling and categorical one-hot encoding
- Linear Regression model in a reproducible scikit-learn pipeline
- Train/test split and MAE, MSE, RMSE, and R² metrics
- Saved model using Joblib
- Interactive Streamlit web application

## Tech stack
Python, Pandas, NumPy, scikit-learn, Matplotlib/Seaborn, Joblib, Streamlit.

## Project structure
```text
Health_Insurance_Premium_Prediction/
├── app.py
├── data/
│   └── insurance.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── Health_Insurance_Premium_Prediction.ipynb
├── src/
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Create and activate a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Explore the notebook
Open the project in VS Code or Jupyter, then open:
`notebooks/Health_Insurance_Premium_Prediction.ipynb`

## 4. Train and evaluate the model
Run from the project root:
```bash
python src/train_model.py
```
This creates `models/insurance_model.joblib` and `models/metrics.json`.

## 5. Launch the app
```bash
streamlit run app.py
```
Open the local URL printed in your terminal (usually `http://localhost:8501`).

## Dataset columns
- `age`: age in years
- `sex`: female/male
- `bmi`: body mass index
- `children`: number of children
- `smoker`: yes/no
- `region`: one of four broad regions
- `charges`: synthetic target value

