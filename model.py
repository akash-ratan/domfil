import joblib
import os
from feature_extraction import extract_features

MODEL_PATH = os.path.join("model", "dns_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)

def predict_domain(domain):
    """
    Predict whether a domain is Benign or Malicious.
    """
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("Please provide a valid domain.")

    domain = domain.strip().lower().rstrip(".")

    features = extract_features(domain)
    prediction = model.predict([features])[0]

    if prediction == 0:
        return "Benign"

    return "Malicious"

def get_prediction_details(domain):
    """
    Return the classification, extracted features,
    and model confidence when available.
    """

    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("Please provide a valid domain.")

    domain = domain.strip().lower().rstrip(".")

    features = extract_features(domain)
    prediction = model.predict([features])[0]

    result = {
        "domain": domain,
        "prediction": (
            "Benign" if prediction == 0 else "Malicious"
        ),
        "features": {
            "length": features[0],
            "digits": features[1],
            "hyphens": features[2],
            "dots": features[3],
            "subdomain_count": features[4],
            "special_characters": features[5],
            "entropy": features[6]
        }
    }

    # Include confidence when the model supports predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        result["confidence"] = round(float(max(probabilities)) * 100, 2)

    return result