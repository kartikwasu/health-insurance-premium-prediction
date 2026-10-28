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

## Upload to GitHub
1. Create a new **public** repository named `health-insurance-premium-prediction`.
2. Extract this ZIP and open the extracted project folder in VS Code.
3. Open the terminal in the project root and run:
```bash
git init
git add .
git commit -m "Add health insurance premium prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/health-insurance-premium-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to authenticate, complete the browser sign-in. Never upload passwords, API keys, or tokens.

## Suggested LinkedIn post
Built an end-to-end **Health Insurance Premium Prediction** demo using Python and Machine Learning.

What I practiced:
- Exploratory Data Analysis
- Data preprocessing and feature encoding
- Linear Regression and model evaluation
- Saving a trained model with Joblib
- Building an interactive Streamlit app

Tech stack: Python | Pandas | scikit-learn | Streamlit

This project uses a synthetic dataset for learning. I'm continuing to improve my skills in Data Science and Machine Learning.

#DataScience #MachineLearning #Python #Streamlit #LearningByDoing

## Next improvements
- Replace the synthetic data with a properly licensed, documented real dataset.
- Compare Linear Regression with Random Forest and Gradient Boosting.
- Add cross-validation, residual plots, and input validation.
- Deploy the app only after checking model quality and clearly documenting limitations.
