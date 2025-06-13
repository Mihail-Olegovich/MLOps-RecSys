#!/usr/bin/env python3
"""
Simple benchmark script for API performance testing.

This script uses curl commands to avoid Python requests library issues
with Docker networking that were causing 502 errors.
"""

import json
import subprocess
import time
from typing import Any


class SimpleBenchmark:
    """Simple benchmark class using curl for HTTP requests."""

    def __init__(self, base_url: str, timeout: int = 15) -> None:
        """Initialize benchmark with base URL and timeout."""
        self.base_url = base_url
        self.timeout = timeout
        self.results: list[dict[str, Any]] = []

    def curl_request(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a curl request and return parsed JSON response."""
        url = f"{self.base_url}{endpoint}"

        if data:
            # POST request with JSON data
            cmd = [
                "curl",
                "-X",
                "POST",
                url,
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(data),
                "--max-time",
                str(self.timeout),
                "--silent",
                "--show-error",
            ]
        else:
            # GET request
            cmd = [
                "curl",
                "-X",
                "GET",
                url,
                "--max-time",
                str(self.timeout),
                "--silent",
                "--show-error",
            ]

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            end_time = time.time()

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Curl failed: {result.stderr}",
                    "response_time": end_time - start_time,
                }

            try:
                response_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": response_data,
                    "response_time": end_time - start_time,
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"JSON decode error: {e}",
                    "response_time": end_time - start_time,
                    "raw_response": result.stdout,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Request timeout",
                "response_time": self.timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
                "response_time": 0,
            }

    def test_health(self) -> dict[str, Any]:
        """Test health endpoint."""
        return self.curl_request("/health")

    def test_recommendation(self, user_id: int, top_k: int = 5) -> dict[str, Any]:
        """Test single user recommendation."""
        return self.curl_request(
            "/api/v1/recommend/user", {"user_id": user_id, "top_k": top_k}
        )

    def run_benchmark(
        self, service_name: str, port: int, user_ids: list[int]
    ) -> dict[str, Any]:
        """Run benchmark for a service."""
        print(f"\nStarting benchmark for {service_name} on port {port}")

        self.base_url = f"http://localhost:{port}"
        times = []
        successes = 0
        failures = 0

        # Test health first
        health_result = self.test_health()
        if not health_result["success"]:
            return {
                "service": service_name,
                "port": port,
                "status": "Service not available",
                "error": health_result.get("error", "Unknown error"),
            }

        print(f"Health check passed for {service_name}")

        # Warm up with first request
        warmup_result = self.test_recommendation(user_ids[0])
        if warmup_result["success"]:
            print(f"Warmup request: {warmup_result['response_time']:.3f}s")

        print(f"Running {len(user_ids)} requests...")

        # Run benchmark requests
        for i, user_id in enumerate(user_ids):
            result = self.test_recommendation(user_id, top_k=5)

            if result["success"]:
                times.append(result["response_time"])
                successes += 1
                if i % 10 == 0:
                    print(f"  Request {i+1}: {result['response_time']:.3f}s")
            else:
                failures += 1
                print(f"  Request {i+1} failed: {result.get('error', 'Unknown')}")

        if times:
            return {
                "service": service_name,
                "port": port,
                "total_requests": len(user_ids),
                "successes": successes,
                "failures": failures,
                "success_rate": successes / len(user_ids),
                "mean_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "p95_time": (
                    sorted(times)[int(0.95 * len(times))]
                    if len(times) > 1
                    else times[0]
                ),
            }
        else:
            return {
                "service": service_name,
                "port": port,
                "status": "All requests failed",
                "failures": failures,
            }

    def compare_services(
        self, services: list[dict[str, Any]], user_ids: list[int]
    ) -> dict[str, Any]:
        """Compare multiple services."""
        results = {}

        for service in services:
            name = service["name"]
            port = service["port"]
            result = self.run_benchmark(name, port, user_ids)
            results[name] = result

        return {
            "comparison": results,
            "summary": self._create_summary(results),
        }

    def _create_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Create summary of benchmark results."""
        summary = {}

        for service, result in results.items():
            if "mean_time" in result:
                summary[service] = {
                    "mean_latency": f"{result['mean_time']:.3f}s",
                    "success_rate": f"{result['success_rate']:.1%}",
                    "p95_latency": f"{result['p95_time']:.3f}s",
                    "total_time": result.get("total_time", 0),
                }

        return summary

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

    def save_results(self, filename: str = "simple_benchmark_results.json") -> None:
        """Save results to JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


def main() -> None:
    """Main benchmark function."""
    # Test data - sample user IDs
    test_user_ids = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

    # Services to test
    services = [
        {"name": "ALS", "port": 8001},
        {"name": "LightFM", "port": 8002},
    ]

    benchmark = SimpleBenchmark("http://localhost:8000", timeout=30)

    print("MLOps RecSys API Benchmark")
    print("Using curl-based requests to avoid Python requests library issues")

    # Run comparison
    results = benchmark.compare_services(services, test_user_ids)

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for service, result in results["comparison"].items():
        print(f"\n{service} Service:")
        if "mean_time" in result:
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Mean Latency: {result['mean_time']:.3f}s")
            print(f"  P95 Latency: {result['p95_time']:.3f}s")
            print(f"  Min/Max: {result['min_time']:.3f}s / {result['max_time']:.3f}s")
        else:
            print(f"  Status: {result.get('status', 'Failed')}")

    # Save results
    benchmark.results = [results]  # Convert dict to list containing the dict
    benchmark.save_results()


if __name__ == "__main__":
    main()
