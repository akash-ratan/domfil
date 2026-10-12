import socket

import dns.message
import dns.query
import dns.rcode

from model import predict_domain


UPSTREAM_DNS = "1.1.1.1"

HOST = "127.0.0.1"
PORT = 1053


def handle_dns_request(data):

    # Convert received bytes into DNS message
    request = dns.message.from_wire(data)

    # Get requested domain
    question = request.question[0]

    domain = str(question.name).rstrip(".")

    print(f"\nDNS Request: {domain}")

    # Ask ML model
    prediction = predict_domain(domain)

    print(f"Prediction: {prediction}")

    # Block malicious domains
    if prediction == "Malicious":

        print("Action: BLOCK")

        response = dns.message.make_response(request)

        response.set_rcode(dns.rcode.NXDOMAIN)

        return response.to_wire()

    # Allow benign domains
    print("Action: ALLOW")

    response = dns.query.udp(
        request,
        UPSTREAM_DNS,
        timeout=5
    )

    return response.to_wire()


# Create UDP socket
server = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

# Bind server
server.bind((HOST, PORT))

print("--------------------------------")
print("AI/ML DNS Filter Started")
print("--------------------------------")
print(f"Listening on {HOST}:{PORT}")
print(f"Upstream DNS: {UPSTREAM_DNS}")
print("--------------------------------")


# Continuously receive DNS requests
while True:

    data, address = server.recvfrom(4096)

    try:

        response = handle_dns_request(data)

        server.sendto(response, address)

    except Exception as e:

        print("Error:", e)