from model import predict_domain


domain = input("Enter domain: ")

result = predict_domain(domain)

print("Domain:", domain)
print("Prediction:", result)