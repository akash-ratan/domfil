import os
import pandas as pd
import joblib

from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from feature_extraction import extract_features


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

data = pd.read_csv(
    "dataset/Kaggle Dataset.csv",
    header=None,
    names=["url", "label", "source"]
)

print("Dataset loaded successfully!")

print("\nFirst five rows:")
print(data.head())


# ==========================================
# STEP 2: CLEAN DATASET
# ==========================================

# Remove rows without URLs or labels
data = data.dropna(subset=["url", "label"]).copy()

# Normalize labels
data["label"] = (
    data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Keep only supported classification labels
valid_labels = ["clean", "malicious"]

data = data[
    data["label"].isin(valid_labels)
].copy()

print("\nDataset after cleaning:", data.shape)


# ==========================================
# STEP 3: EXTRACT DOMAIN NAMES
# ==========================================

def get_domain(url):

    url = str(url).strip()

    if not url:
        return None

    # Add a scheme if the URL does not contain one
    if "://" not in url:
        url = "http://" + url

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname

        if domain:
            return domain.lower().rstrip(".")

        return None

    except ValueError:
        return None


data["domain"] = data["url"].apply(get_domain)

# Remove invalid URLs
data = data.dropna(subset=["domain"]).copy()


# ==========================================
# STEP 4: MAP LABELS
# ==========================================

label_mapping = {
    "clean": 0,
    "malicious": 1
}

data["label"] = data["label"].map(label_mapping)

data = data.dropna(subset=["label"]).copy()

data["label"] = data["label"].astype(int)


# ==========================================
# STEP 5: DISPLAY DATASET INFORMATION
# ==========================================

print("\nFinal dataset shape:", data.shape)

print("\nLabel distribution:")
print(
    data["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel meanings:")
print("0 = Clean / Benign")
print("1 = Malicious, including phishing URLs labelled malicious")

print("\nSource distribution:")
print(
    data["source"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
    .value_counts()
)


# ==========================================
# STEP 6: CHECK DATASET
# ==========================================

if data["label"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both Clean and Malicious labels."
    )

# Remove duplicate domains to reduce train/test leakage.
# If a domain appears with conflicting labels, exclude it.
label_counts = data.groupby("domain")["label"].nunique()

conflicting_domains = label_counts[
    label_counts > 1
].index

if len(conflicting_domains) > 0:
    print(
        "\nRemoving domains with conflicting labels:",
        len(conflicting_domains)
    )

    data = data[
        ~data["domain"].isin(conflicting_domains)
    ].copy()

# Keep one row per domain
data = data.drop_duplicates(
    subset=["domain"],
    keep="first"
).copy()

print("\nDataset shape after domain deduplication:", data.shape)

if data["label"].nunique() != 2:
    raise ValueError(
        "Both classes must remain after removing duplicate domains."
    )

if data["label"].value_counts().min() < 2:
    raise ValueError(
        "Each class needs at least two domains for stratified splitting."
    )


# ==========================================
# STEP 7: EXTRACT FEATURES
# ==========================================

print("\nExtracting domain features...")

X = []

for domain in data["domain"]:
    features = extract_features(domain)
    X.append(features)

y = data["label"]

print("Feature extraction completed.")
print("Features per domain:", len(X[0]))


# ==========================================
# STEP 8: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# STEP 9: TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# STEP 10: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# STEP 11: EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# STEP 12: CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Clean / Benign",
            "Malicious / Phishing"
        ],
        zero_division=0
    )
)


# ==========================================
# STEP 13: CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Interpretation:")
print("True Negatives  :", cm[0][0])
print("False Positives :", cm[0][1])
print("False Negatives :", cm[1][0])
print("True Positives  :", cm[1][1])


# ==========================================
# STEP 14: SAVE TRAINED MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/dns_model.pkl")

print("\nModel saved successfully!")
print("Location: model/dns_model.pkl")