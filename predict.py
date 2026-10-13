import webbrowser
import time

from urllib.parse import urlparse
from model import predict_domain
from dns_resolver import resolve_domain


def extract_domain(url):
    """Extract the hostname from a URL."""

    url = url.strip()

    if not url:
        return None

    if "://" not in url:
        url = "https://" + url

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname

        if domain:
            return domain.lower().rstrip(".")

        return None

    except ValueError:
        return None


def predict_website():
    print("=" * 50)
    print("       AI/ML-BASED DNS FIREWALL")
    print("=" * 50)

    url = input("\nEnter website URL: ").strip()

    if not url:
        print("Error: Please enter a valid website URL.")
        return

    domain = extract_domain(url)

    if not domain:
        print("Error: Invalid website URL.")
        return

    print(f"\nWebsite: {url}")
    print(f"Domain: {domain}")

    print("\n[1] Machine Learning Analysis")
    print("-" * 35)

    try:
        prediction = predict_domain(domain)
    except Exception as error:
        print(f"ML prediction failed: {error}")
        print("Website will not be opened.")
        return

    print(f"ML Prediction: {prediction}")

    if str(prediction).strip().lower() == "malicious":
        print("\nFINAL RESULT: MALICIOUS WEBSITE")
        print("Action: BLOCKED")
        print("The website will not be opened.")
        return

    print("\n[2] DNS Firewall Analysis")
    print("-" * 35)

    try:
        dns_result = resolve_domain(domain)
    except Exception as error:
        print(f"DNS analysis failed: {error}")
        print("Website will not be opened.")
        return

    dns_status = dns_result.get("status", "Unknown")
    ip_addresses = dns_result.get("ip_addresses", [])

    print(f"DNS Status: {dns_status}")

    if ip_addresses:
        print(f"IP Addresses: {ip_addresses}")

    if dns_status != "Resolved":
        print("\nFINAL RESULT: WEBSITE NOT OPENED")
        print("DNS resolution failed or the domain could not be resolved.")
        print("Check whether the DNS firewall is running.")
        return

    print("\nFINAL RESULT: ML-BENIGN AND DNS-RESOLVED")
    print("Action: ALLOW")
    print("Opening website in 2 seconds...")
    time.sleep(2)
    webbrowser.open(url)

    if "://" not in url:
        url = "https://" + url

    webbrowser.open(url)

if __name__ == "__main__":
    try:
        predict_website()
    except KeyboardInterrupt:
        print("\nProgram terminated.")