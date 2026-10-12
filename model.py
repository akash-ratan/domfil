import joblib
from feature_extraction import extract_features

model = joblib.load("model/dns_model.pkl")

def predict_domain(domain):
    features = extract_features(domain)
    prediction = model.predict([features])[0]

    if prediction == 0: return "Benign"
    else: return "Malicious"
    