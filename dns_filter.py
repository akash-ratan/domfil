import socket, dns.message, dns.query, dns.rcode
from model import predict_domain

UPSTREAM_DNS = "1.1.1.1"

HOST = "127.0.0.1"
PORT = 1053

def handle_dns_request(data):
    request = dns.message.from_wire(data)
    question = request.question[0]
    domain = str(question.name).rstrip(".")

    print(f"\nDNS Request: {domain}")

    prediction = predict_domain(domain)

    print(f"Prediction: {prediction}")

    if prediction == "Malicious":
        print("Action: BLOCK")
        response = dns.message.make_response(request)
        response.set_rcode(dns.rcode.NXDOMAIN)

        return response.to_wire()

    print("Action: ALLOW")
    response = dns.query.udp(request, UPSTREAM_DNS, timeout=5)

    return response.to_wire()

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))

print("--------------------------------")
print("AI/ML DNS Filter Started")
print("--------------------------------")
print(f"Listening on {HOST}:{PORT}")
print(f"Upstream DNS: {UPSTREAM_DNS}")
print("--------------------------------")

while True:
    data, address = server.recvfrom(4096)

    try:
        response = handle_dns_request(data)
        server.sendto(response, address)

    except Exception as e: print("Error:", e)
    