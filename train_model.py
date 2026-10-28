from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"
MODEL_PATH = ROOT / "models" / "insurance_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

def main():
    df = pd.read_csv(DATA_PATH)
    target = "charges"
    X = df.drop(columns=[target])
    y = df[target]

    categorical = ["sex", "smoker", "region"]
    numeric = ["age", "bmi", "children"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "mse": round(float(mean_squared_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "dataset_note": "Synthetic demonstration dataset; metrics are for demonstration only."
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model saved to:", MODEL_PATH)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
