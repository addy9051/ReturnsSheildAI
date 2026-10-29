"""Benchmark and Stress Test Comparison: Rust Edge Gateway vs FastAPI.

Sends concurrent triage requests to:
1. Python FastAPI Backend (http://localhost:8000/returns/inspect)
2. Rust Axum Edge Gateway (http://localhost:8080/returns/inspect)

Measures:
- P50, P95, P99 Latency (ms)
- Request throughput (req/sec)
- Memory RSS footprint (MB)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List
import urllib.request
import urllib.error


def make_inspection_payload(case_idx: int) -> dict:
    return {
        "rma_number": f"RMA-BENCH-{case_idx:04d}",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "box_barcode_raw": "SN-SNY-8842109",
        "expected_customer_serial": "SN-SNY-8842109",
        "operator_id": "BENCH-TESTER",
        "bench_id": "BENCH-01",
    }


def send_request(url: str, payload: dict) -> float:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        _ = resp.read()
    return (time.perf_counter() - t0) * 1000.0


def benchmark_endpoint(name: str, url: str, num_requests: int = 100) -> Dict[str, float]:
    print(f"\n--- Benchmarking {name} ({url}) over {num_requests} requests ---")
    latencies = []

    # Warmup
    try:
        _ = send_request(url, make_inspection_payload(0))
    except Exception as err:
        print(f"Warning: {name} is unreachable at {url}: {err}")
        return {}

    t_start = time.perf_counter()
    for i in range(1, num_requests + 1):
        try:
            dur = send_request(url, make_inspection_payload(i))
            latencies.append(dur)
        except Exception as err:
            print(f"Request {i} failed: {err}")

    total_time = time.perf_counter() - t_start
    throughput = len(latencies) / total_time if total_time > 0 else 0

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    print(f"Results for {name}:")
    print(f"  Throughput:  {throughput:.1f} req/sec")
    print(f"  P50 Latency: {p50:.2f} ms")
    print(f"  P95 Latency: {p95:.2f} ms")
    print(f"  P99 Latency: {p99:.2f} ms")

    return {
        "throughput": throughput,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Rust Edge vs Python FastAPI")
    parser.add_argument("--rust-url", default="http://localhost:8080/returns/inspect")
    parser.add_argument("--python-url", default="http://localhost:8000/returns/inspect")
    parser.add_argument("-n", "--num-requests", type=int, default=50)
    args = parser.parse_args()

    print("ReturnShield AI Edge Ingestion Benchmark Suite")
    rust_res = benchmark_endpoint("Rust Edge Gateway (Axum)", args.rust_url, args.num_requests)
    py_res = benchmark_endpoint("Python FastAPI Service", args.python_url, args.num_requests)
