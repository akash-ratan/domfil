import socket
import dns.message,dns.query,dns.rcode,dns.flags,dns.exception

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from model import predict_domain
UPSTREAM_DNS = "8.8.8.8"

HOST = "127.0.0.1"
PORT = 1053

DNS_TIMEOUT = 3
MAX_WORKERS = 20
BUFFER_SIZE = 4096

def log_event(domain, prediction, action, details=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"[{timestamp}] "
        f"Domain: {domain} | "
        f"Prediction: {prediction} | "
        f"Action: {action}"
    )

    if details:
        message += f" | Details: {details}"

    print(message)

def handle_dns_request(data):

    request = None
    domain = "Unknown"

    try:
        request = dns.message.from_wire(data)
        if request.flags & dns.flags.QR:
            return None
        if len(request.question) != 1:
            return None

        question = request.question[0]

        domain = str(question.name).rstrip(".").lower()
        query_type = dns.rdatatype.to_text(question.rdtype)

        if query_type not in ("A", "AAAA"):
            log_event(
                domain,
                "Not checked",
                "REFUSED",
                f"Unsupported query type: {query_type}"
            )

            response = dns.message.make_response(request)
            response.set_rcode(dns.rcode.REFUSED)

            return response.to_wire()

        prediction = predict_domain(domain)

        log_event(
            domain,
            prediction,
            "Checking"
        )

        if prediction.strip().lower() == "malicious":

            log_event(
                domain,
                prediction,
                "BLOCKED"
            )

            response = dns.message.make_response(request)
            response.set_rcode(dns.rcode.NXDOMAIN)

            return response.to_wire()
        
        response = dns.query.udp(
            request,
            UPSTREAM_DNS,
            timeout=DNS_TIMEOUT
        )

        log_event(
            domain,
            prediction,
            "ALLOWED",
            f"Query type: {query_type}"
        )

        return response.to_wire()

    except (dns.exception.DNSException, OSError) as error:

        log_event(
            domain,
            "Unknown",
            "ERROR",
            str(error)
        )

        if request is not None:
            response = dns.message.make_response(request)
            response.set_rcode(dns.rcode.SERVFAIL)

            return response.to_wire()

        return None

    except Exception as error:

        log_event(
            domain,
            "Unknown",
            "ERROR",
            str(error)
        )

        if request is not None:
            response = dns.message.make_response(request)
            response.set_rcode(dns.rcode.SERVFAIL)

            return response.to_wire()

        return None

def main():

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    server.bind((HOST, PORT))

    print("=" * 50)
    print("       AI/ML DNS FIREWALL")
    print("=" * 50)
    print(f"Listening on: {HOST}:{PORT}")
    print(f"Upstream DNS: {UPSTREAM_DNS}")
    print(f"DNS timeout: {DNS_TIMEOUT} seconds")
    print(f"Maximum workers: {MAX_WORKERS}")
    print("=" * 50)

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        try:
            while True:

                data, address = server.recvfrom(BUFFER_SIZE)

                future = executor.submit(
                    handle_dns_request,
                    data
                )

                def send_response(future, address=address):
                    try:
                        response = future.result()

                        if response is not None:
                            server.sendto(response, address)

                    except Exception as error:
                        print("Response error:", error)

                future.add_done_callback(send_response)

        except KeyboardInterrupt:
            print("\nDNS firewall shutting down.")

        finally:
            server.close()

if __name__ == "__main__":
    main()