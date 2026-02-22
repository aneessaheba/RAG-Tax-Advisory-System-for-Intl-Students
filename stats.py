"""
Production Stats — reads query_log.jsonl and prints real metrics:
  - Total queries
  - Mean latency
  - p95 latency  (95th percentile of total_latency_s)
  - Fallback rate (fraction of queries where Gemini failed and extractive fallback was used)

Run this after using the chatbot (app.py) to see actual production numbers.
"""
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(__file__)
QUERY_LOG_PATH = os.path.join(BASE_DIR, 'query_log.jsonl')


def load_log():
    if not os.path.exists(QUERY_LOG_PATH):
        print("No query_log.jsonl found. Run app.py first and ask some questions.")
        return []
    entries = []
    with open(QUERY_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    entries = load_log()
    if not entries:
        return

    n = len(entries)
    latencies = [e["total_latency_s"] for e in entries]
    retrieval_latencies = [e["retrieval_latency_s"] for e in entries]
    llm_latencies = [e["llm_latency_s"] for e in entries]
    fallbacks = [e["used_fallback"] for e in entries]
    confidences = [e["confidence"] for e in entries]

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fallback_rate = sum(fallbacks) / n

    print("=" * 50)
    print("  Production Stats from query_log.jsonl")
    print("=" * 50)
    print(f"  Total queries logged : {n}")
    print()
    print("  Latency (end-to-end):")
    print(f"    Mean              : {np.mean(latencies):.2f}s")
    print(f"    p50 (median)      : {p50:.2f}s")
    print(f"    p95               : {p95:.2f}s")
    print(f"    p99               : {p99:.2f}s")
    print(f"    Max               : {max(latencies):.2f}s")
    print()
    print("  Latency breakdown (mean):")
    print(f"    Retrieval         : {np.mean(retrieval_latencies):.2f}s")
    print(f"    LLM (Gemini)      : {np.mean(llm_latencies):.2f}s")
    print()
    print("  Retrieval confidence:")
    print(f"    Mean              : {np.mean(confidences):.3f}")
    print(f"    Min               : {min(confidences):.3f}")
    print()
    print(f"  Fallback rate       : {fallback_rate:.1%}  ({sum(fallbacks)}/{n} queries used extractive fallback)")
    print("=" * 50)


if __name__ == "__main__":
    main()
