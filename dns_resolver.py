import dns.resolver

def resolve_domain(domain):
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ["127.0.0.1"]
    resolver.port = 1053
    resolver.timeout = 3
    resolver.lifetime = 5

    result = {
        "domain": domain,
        "status": "Unknown",
        "ip_addresses": []
    }

    try:
        answers = resolver.resolve(domain, "A")

        result["ip_addresses"] = [
            str(answer) for answer in answers
        ]
        result["status"] = "Resolved"

    except dns.resolver.NXDOMAIN:
        result["status"] = "Blocked / Domain Not Found"

    except dns.resolver.NoAnswer:
        result["status"] = "No A Record"

    except dns.resolver.Timeout:
        result["status"] = "DNS Timeout"

    except dns.exception.DNSException:
        result["status"] = "DNS Error"

    return result