import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from feature_extraction import extract_features


# 1. Load dataset
data = pd.read_csv("dataset/domains.csv")

print("Dataset loaded successfully.")
print(data.head())


# 2. Separate domains and labels
domains = data["domain"]
labels = data["label"]


# 3. Extract features
X = []

for domain in domains:
    features = extract_features(domain)
    X.append(features)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 6. Train model
model.fit(X_train, y_train)


# 7. Predict test data
y_pred = model.predict(X_test)


# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


print("\nModel Performance")
print("-----------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# 9. Save model
joblib.dump(model, "model/dns_model.pkl")

print("\nModel saved successfully!")