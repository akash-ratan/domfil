import re
import math
from collections import Counter

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "password",
    "account",
    "banking",
    "secure",
    "update",
    "confirm",
    "wallet",
    "bonus",
    "free",
    "support"
]

def normalize_domain(domain):
    """
    Normalize a hostname before feature extraction.
    """

    if not isinstance(domain, str):
        raise ValueError("Domain must be a string.")

    domain = domain.strip().lower().rstrip(".")

    if not domain:
        raise ValueError("Domain cannot be empty.")

    try:
        domain = domain.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise ValueError("Invalid internationalized domain.") from error

    return domain


def calculate_entropy(domain):
    """
    Calculate Shannon character entropy.
    """

    if not domain:
        return 0.0

    counter = Counter(domain)
    length = len(domain)

    entropy = 0.0

    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def extract_features(domain):
    """
    Extract 15 lexical features from a domain.

    IMPORTANT:
    Input must be a hostname, not a complete URL.
    """

    domain = normalize_domain(domain)

    length = len(domain)

    digits = sum(c.isdigit() for c in domain)
    hyphens = domain.count("-")
    dots = domain.count(".")

    special_chars = len(
        re.findall(r"[^a-zA-Z0-9.-]", domain)
    )

    labels = domain.split(".")

    number_of_labels = len(labels)

    extra_label_count = max(0, number_of_labels - 2)

    label_lengths = [len(label) for label in labels]

    longest_label = max(label_lengths, default=0)

    average_label_length = (
        sum(label_lengths) / len(label_lengths)
        if label_lengths else 0.0
    )

    digit_ratio = digits / length if length else 0.0

    hyphen_ratio = hyphens / length if length else 0.0

    unique_character_ratio = (
        len(set(domain)) / length
        if length else 0.0
    )

    max_repeated_run = 0
    current_run = 0
    previous_char = None

    for char in domain:
        if char == previous_char:
            current_run += 1
        else:
            current_run = 1
            previous_char = char

        max_repeated_run = max(
            max_repeated_run,
            current_run
        )

    keyword_count = 0

    for label in labels:
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in label:
                keyword_count += 1

    entropy = calculate_entropy(domain)

    features = [
        length,
        digits,
        hyphens,
        dots,
        extra_label_count,
        special_chars,
        entropy,
        number_of_labels,
        longest_label,
        average_label_length,
        digit_ratio,
        hyphen_ratio,
        unique_character_ratio,
        max_repeated_run,
        keyword_count
    ]

    return features