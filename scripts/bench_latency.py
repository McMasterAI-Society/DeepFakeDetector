"""
Latency benchmark for the DeepFake Detector API.

Measures, against an already running API (local or the HF Space):

1. Cold start (optional, --cold): polls GET /ready from the moment the script
   starts and records time to first HTTP response and time to "ready".
   Start the script right after triggering a restart / while the Space sleeps.
2. First request after ready (explain off): shows whether warm-up moved CUDA
   init and cuDNN autotuning off the user path.
3. Warm /predict latency, explain off and explain on: p50 / p95 / mean of the
   client-side wall time plus the server-reported timing_ms.

Results are written as JSON so before/after runs can be compared.

Usage:
    python scripts/bench_latency.py --base-url http://localhost:8000
    python scripts/bench_latency.py --base-url https://<user>-<space>.hf.space \
        --cold --n 30 --label slim-image --out bench-slim.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "frontend" / "src" / "assets" / "demo" / "gemini-d001.jpg"


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def wait_for_ready(
    session: requests.Session,
    base_url: str,
    timeout_s: float,
    interval_s: float,
) -> Dict[str, Any]:
    """Poll /ready until it reports ready; return cold-start timings (seconds)."""
    t0 = time.perf_counter()
    first_response_s: Optional[float] = None
    first_not_ready_s: Optional[float] = None
    polls = 0
    last_error: Optional[str] = None

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > timeout_s:
            return {
                "ready": False,
                "time_to_first_response_s": first_response_s,
                "time_to_ready_s": None,
                "polls": polls,
                "last_error": last_error,
            }
        polls += 1
        try:
            resp = session.get(f"{base_url}/ready", timeout=min(30.0, timeout_s))
            if resp.status_code == 200:
                if first_response_s is None:
                    first_response_s = time.perf_counter() - t0
                body = resp.json()
                if body.get("status") == "ready":
                    return {
                        "ready": True,
                        "time_to_first_response_s": round(first_response_s, 2),
                        "time_to_ready_s": round(time.perf_counter() - t0, 2),
                        "first_not_ready_s": first_not_ready_s,
                        "polls": polls,
                    }
                if first_not_ready_s is None:
                    first_not_ready_s = round(time.perf_counter() - t0, 2)
            else:
                last_error = f"HTTP {resp.status_code}"
        except (requests.RequestException, ValueError) as e:
            last_error = type(e).__name__
        time.sleep(interval_s)


def predict_once(
    session: requests.Session,
    base_url: str,
    image_bytes: bytes,
    image_name: str,
    explain: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    params = {"use_fusion": "true", "return_submodels": "true", "explain": str(explain).lower()}
    files = {"image": (image_name, image_bytes, "application/octet-stream")}
    t0 = time.perf_counter()
    resp = session.post(f"{base_url}/predict", params=params, files=files, timeout=timeout_s)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    timing = body.get("timing_ms") or {}
    return {
        "wall_ms": wall_ms,
        "server_total_ms": timing.get("total"),
        "server_inference_ms": timing.get("inference"),
        "prob_fake": (body.get("final") or {}).get("prob_fake"),
    }


def bench_predict(
    session: requests.Session,
    base_url: str,
    image_bytes: bytes,
    image_name: str,
    explain: bool,
    n: int,
    warmup: int,
    timeout_s: float,
) -> Dict[str, Any]:
    for _ in range(warmup):
        predict_once(session, base_url, image_bytes, image_name, explain, timeout_s)

    samples = [
        predict_once(session, base_url, image_bytes, image_name, explain, timeout_s)
        for _ in range(n)
    ]
    probs = {s["prob_fake"] for s in samples if s["prob_fake"] is not None}
    return {
        "explain": explain,
        "warmup_requests": warmup,
        "wall_ms": summarize([s["wall_ms"] for s in samples]),
        "server_total_ms": summarize(
            [s["server_total_ms"] for s in samples if s["server_total_ms"] is not None]
        ),
        "server_inference_ms": summarize(
            [s["server_inference_ms"] for s in samples if s["server_inference_ms"] is not None]
        ),
        "prob_fake_values": sorted(probs),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Image to upload")
    parser.add_argument("--n", type=int, default=20, help="Timed requests per mode")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed requests per mode before timing")
    parser.add_argument("--cold", action="store_true", help="Measure time-to-ready by polling /ready first")
    parser.add_argument("--ready-timeout", type=float, default=900.0, help="Seconds to wait for /ready")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between /ready polls")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Per-request timeout (s)")
    parser.add_argument("--skip-explain", action="store_true", help="Only benchmark explain=false")
    parser.add_argument("--label", default="", help="Free-text variant label stored in the JSON")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    image_bytes = args.image.read_bytes()
    session = requests.Session()

    result: Dict[str, Any] = {
        "label": args.label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "git_commit": git_commit(),
        "client": {"python": sys.version.split()[0], "platform": platform.platform()},
        "image": {
            "path": str(args.image),
            "bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "params": {"n": args.n, "warmup": args.warmup},
    }

    if args.cold:
        print(f"Polling {base_url}/ready (timeout {args.ready_timeout:.0f}s)...")
        cold = wait_for_ready(session, base_url, args.ready_timeout, args.poll_interval)
        result["cold_start"] = cold
        print(f"  cold start: {cold}")
        if not cold["ready"]:
            print("API never became ready; writing partial results.")
            return _write(result, args.out, exit_code=1)
    else:
        cold_check = wait_for_ready(session, base_url, 60.0, args.poll_interval)
        if not cold_check["ready"]:
            print("API is not ready (use --cold to wait longer).")
            result["error"] = "not_ready"
            return _write(result, args.out, exit_code=1)

    # First request after ready: only meaningful right after a (re)start.
    first = predict_once(session, base_url, image_bytes, args.image.name, False, args.request_timeout)
    result["first_request_after_ready"] = {
        "explain": False,
        "wall_ms": round(first["wall_ms"], 2),
        "server_total_ms": first["server_total_ms"],
        "note": "only a cold-path measurement when run with --cold right after a restart",
    }
    print(f"  first request after ready: {result['first_request_after_ready']}")

    modes = [False] if args.skip_explain else [False, True]
    result["predict"] = {}
    for explain in modes:
        key = "explain_on" if explain else "explain_off"
        print(f"Benchmarking /predict {key} (n={args.n}, warmup={args.warmup})...")
        stats = bench_predict(
            session, base_url, image_bytes, args.image.name, explain,
            args.n, args.warmup, args.request_timeout,
        )
        result["predict"][key] = stats
        print(f"  wall_ms: {stats['wall_ms']}")
        print(f"  server_total_ms: {stats['server_total_ms']}")

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return _write(result, args.out, exit_code=0)


def _write(result: Dict[str, Any], out: Optional[Path], exit_code: int) -> int:
    text = json.dumps(result, indent=2)
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(f"bench-latency-{stamp}.json")
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
