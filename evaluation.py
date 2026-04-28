import re
import numpy as np


def extract_claims(answer: str):
    sentences = re.split(r'[.?!]\s+', answer)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def score_claims(claims, docs, reranker):
    results = []

    for claim in claims:
        pairs = [(claim, doc.page_content) for doc in docs]
        scores = reranker.model.predict(pairs)

        max_score = float(np.max(scores)) if len(scores) > 0 else -999

        results.append({
            "claim": claim,
            "max_score": max_score
        })

    return results


def compute_coverage(claim_scores, threshold=-5.5):
    supported = [c for c in claim_scores if c["max_score"] > threshold]

    total = len(claim_scores)
    supported_count = len(supported)

    coverage = supported_count / total if total > 0 else 0.0

    return {
        "coverage": coverage,
        "supported": supported_count,
        "total": total
    }