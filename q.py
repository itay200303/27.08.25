import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = {
    "Age": [25, 38, 29, 47, 35, 53, 31, 42, 40, 50],
    "Past_Claims": [0, 2, 1, 3, 1, 4, 0, 2, 2, 3],
    "Insurance_Amount": [120, 200, 150, 300, 220, 400, 130, 180, 250, 350],
    "Compensation?": ["No", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"]
}

df = pd.DataFrame(data)

X = df[["Age", "Past_Claims", "Insurance_Amount"]]
y = df["Compensation?"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("OOB Accuracy:", model.oob_score_)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

joblib.dump(model, "random_forest_model.pkl")

loaded_model = joblib.load("random_forest_model.pkl")
print("Prediction for first test sample:", loaded_model.predict([X_test.iloc[0]]))