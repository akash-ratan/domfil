import re
import math
from collections import Counter

def calculate_entropy(domain):
    counter = Counter(domain)
    length = len(domain)

    entropy = 0

    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy

def extract_features(domain):
    length=len(domain)
    digits=sum(c.isdigit() for c in domain)
    hypens=domain.count("-")
    dots=domain.count(".")
    sumdomains=max(0,dots-1)
    special_chars=len(re.findall(r"[^a-zA-Z0-9.-]",domain))
    entropy=calculate_entropy(domain)
    features=[
        length,
        digits,
        hypens,
        dots,
        sumdomains,
        special_chars,
        entropy
    ]
    return features
