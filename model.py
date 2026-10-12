import joblib

from feature_extraction import extract_features


# Load trained model
model = joblib.load("model/dns_model.pkl")


def predict_domain(domain):

    # Extract features
    features = extract_features(domain)

    # Make prediction
    prediction = model.predict([features])[0]

    # Convert prediction to readable result
    if prediction == 0:
        return "Benign"
    else:
        return "Malicious"