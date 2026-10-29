"""Benchmark and Stress Test Comparison: Rust Edge Gateway vs FastAPI.

Sends concurrent triage requests to:
1. Python FastAPI Backend (http://127.0.0.1:8000/returns/inspect)
2. Rust Axum Edge Gateway (http://127.0.0.1:8080/returns/inspect)

Measures:
- P50, P95, P99 Latency (ms)
- Request throughput (req/sec)
- High concurrency under ThreadPoolExecutor with HTTP connection pooling (httpx)
- Process Memory RSS footprint (MB)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time
from typing import Dict, List, Optional

import httpx
import psutil


def make_inspection_payload(case_idx: int) -> dict:
    return {
        "rma_number": f"RMA-BENCH-{case_idx:04d}",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "box_barcode_raw": "SN-SNY-8842109",
        "expected_customer_serial": "SN-SNY-8842109",
        "operator_id": "BENCH-TESTER",
        "bench_id": "BENCH-01",
    }


def get_process_memory_mb(keyword: str) -> float:
    """Finds matching process RSS memory in MB."""
    rss_total = 0.0
    found = False
    for proc in psutil.process_iter(["name", "memory_info", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])
            if keyword.lower() in name.lower() or keyword.lower() in cmdline.lower():
                rss_total += proc.info["memory_info"].rss / (1024 * 1024)
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return round(rss_total, 2) if found else 0.0


def benchmark_endpoint(
    name: str,
    url: str,
    num_requests: int = 500,
    concurrency: int = 20,
    proc_keyword: Optional[str] = None,
) -> Dict[str, float]:
    print(f"\n========================================================")
    print(f"Benchmarking {name}")
    print(f"URL: {url} | Total Requests: {num_requests} | Concurrency: {concurrency}")
    print(f"========================================================")

    # Establish HTTP client with persistent connection pool
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    client = httpx.Client(limits=limits, timeout=10.0)

    # Warmup passes
    try:
        for _ in range(5):
            res = client.post(url, json=make_inspection_payload(0))
            res.raise_for_status()
    except Exception as err:
        print(f"ERROR: {name} is unreachable at {url}: {err}")
        client.close()
        return {}

    latencies: List[float] = []
    errors = 0

    def worker_request(case_idx: int) -> float:
        payload = make_inspection_payload(case_idx)
        t0 = time.perf_counter()
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return (time.perf_counter() - t0) * 1000.0

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker_request, i)
            for i in range(1, num_requests + 1)
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                lat = f.result()
                latencies.append(lat)
            except Exception as err:
                errors += 1

    total_time = time.perf_counter() - t_start
    throughput = len(latencies) / total_time if total_time > 0 else 0
    client.close()

    latencies.sort()
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
    avg = statistics.mean(latencies) if latencies else 0.0

    mem_mb = get_process_memory_mb(proc_keyword) if proc_keyword else 0.0

    print(f"Results for {name}:")
    print(f"  Successful:   {len(latencies)} / {num_requests} ({errors} errors)")
    print(f"  Total Time:   {total_time:.3f} s")
    print(f"  Throughput:   {throughput:.1f} req/sec")
    print(f"  Mean Latency: {avg:.2f} ms")
    print(f"  P50 Latency:  {p50:.2f} ms")
    print(f"  P95 Latency:  {p95:.2f} ms")
    print(f"  P99 Latency:  {p99:.2f} ms")
    if mem_mb > 0:
        print(f"  Memory RSS:   {mem_mb:.1f} MB")

    return {
        "throughput": throughput,
        "mean": avg,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "memory_mb": mem_mb,
        "success": len(latencies),
        "errors": errors,
    }


def print_comparison_table(rust: Dict[str, float], py: Dict[str, float]):
    if not rust or not py:
        return

    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Python FastAPI':<18} | {'Rust Axum Gateway':<18} | {'Advantage'}")
    print("-" * 70)

    tp_mult = rust["throughput"] / py["throughput"] if py["throughput"] > 0 else 0
    p50_mult = py["p50"] / rust["p50"] if rust["p50"] > 0 else 0
    p95_mult = py["p95"] / rust["p95"] if rust["p95"] > 0 else 0
    p99_mult = py["p99"] / rust["p99"] if rust["p99"] > 0 else 0

    print(f"{'Throughput (req/s)':<25} | {py['throughput']:>14.1f} req/s | {rust['throughput']:>14.1f} req/s | {tp_mult:>6.1f}x higher")
    print(f"{'P50 Latency (ms)':<25} | {py['p50']:>17.2f} ms | {rust['p50']:>17.2f} ms | {p50_mult:>6.1f}x faster")
    print(f"{'P95 Latency (ms)':<25} | {py['p95']:>17.2f} ms | {rust['p95']:>17.2f} ms | {p95_mult:>6.1f}x faster")
    print(f"{'P99 Latency (ms)':<25} | {py['p99']:>17.2f} ms | {rust['p99']:>17.2f} ms | {p99_mult:>6.1f}x faster")
    if rust["memory_mb"] > 0 and py["memory_mb"] > 0:
        mem_ratio = py["memory_mb"] / rust["memory_mb"]
        print(f"{'Memory Footprint (RSS)':<25} | {py['memory_mb']:>17.1f} MB | {rust['memory_mb']:>17.1f} MB | {mem_ratio:>6.1f}x leaner")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Rust Edge vs Python FastAPI")
    parser.add_argument("--rust-url", default="http://127.0.0.1:8080/returns/inspect")
    parser.add_argument("--python-url", default="http://127.0.0.1:8000/returns/inspect")
    parser.add_argument("-n", "--num-requests", type=int, default=500)
    parser.add_argument("-c", "--concurrency", type=int, default=20)
    args = parser.parse_args()

    print("ReturnShield AI Concurrent Ingestion Benchmark Suite (HTTP Keep-Alive)")
    rust_res = benchmark_endpoint(
        "Rust Edge Gateway (Axum)",
        args.rust_url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        proc_keyword="returnshield-edge",
    )
    py_res = benchmark_endpoint(
        "Python FastAPI Service",
        args.python_url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        proc_keyword="uvicorn",
    )

    print_comparison_table(rust_res, py_res)
