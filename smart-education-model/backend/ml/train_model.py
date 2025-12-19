import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_df

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "..", "data", "StudentsPerformance.csv")
MODEL_PATH = os.path.join(BASE, "model.pkl")

def train_model():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_df(df)

    X = df[["attendance","avg_marks","assignment_completion","behavior_score"]]
    y = df["risk_level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
