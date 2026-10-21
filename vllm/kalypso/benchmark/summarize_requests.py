"""Summarize an instrumentation JSONL file (see vllm/kalypso/instrumentation.py).

    python vllm/kalypso/benchmark/summarize_requests.py runs/requests.jsonl [--plot out.png]

Prints per-worker request counts, latency p50/p99, token totals, throughput and
peak queue depth. With --plot (needs matplotlib) also saves a 3-panel chart of
request counts, latency and throughput per worker.
"""
import argparse
import json
import statistics
from collections import defaultdict


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def load(path):
    reqs, samples = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["type"] == "request":
                reqs.append(row)
            elif row["type"] == "queue_sample":
                samples.append(row)
    return reqs, samples


def summarize(reqs, samples):
    ok = [r for r in reqs if r["status"] == "ok"]
    if not ok:
        return {}
    window = max(r["finish_t"] for r in ok) - min(r["submit_t"] for r in ok)
    by = defaultdict(list)
    for r in ok:
        by[r["worker"]].append(r)
    depth = defaultdict(float)
    for s in samples:
        depth[s["worker"]] = max(depth[s["worker"]], s["waiting"] + s["running"])
    out = {}
    for w, rs in sorted(by.items()):
        lat = [r["finish_t"] - r["submit_t"] for r in rs]
        out[w] = {
            "requests": len(rs),
            "errors": sum(1 for r in reqs if r["worker"] == w and r["status"] != "ok"),
            "lat_p50_s": _pct(lat, 0.50),
            "lat_p99_s": _pct(lat, 0.99),
            "lat_mean_s": statistics.fmean(lat),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs),
            "throughput_rps": len(rs) / window if window > 0 else float("nan"),
            "peak_queue_depth": depth.get(w),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    reqs, samples = load(a.path)
    stats = summarize(reqs, samples)
    for w, s in stats.items():
        print(w, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    if a.plot and stats:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ws = list(stats)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        ax[0].bar(ws, [stats[w]["requests"] for w in ws]); ax[0].set_title("Requests per worker")
        ax[1].bar(ws, [stats[w]["lat_p50_s"] for w in ws], label="p50")
        ax[1].bar(ws, [stats[w]["lat_p99_s"] for w in ws], alpha=0.5, label="p99")
        ax[1].set_title("Request latency (s)"); ax[1].legend()
        ax[2].bar(ws, [stats[w]["throughput_rps"] for w in ws]); ax[2].set_title("Throughput (req/s)")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
