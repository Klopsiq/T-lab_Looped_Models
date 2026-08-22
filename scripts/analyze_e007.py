"""Analyze the E007 seed replication using the preregistered rules."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E007"
E006 = ROOT / "reports" / "E006" / "raw" / "runs" / "E006"
E007 = REPORT / "raw" / "runs" / "E007"
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
FINAL = "combined_random8_24_final"
AUX = "combined_random8_24_aux"
SEEDS = [23, 47, 71, 89]
NEW_SEEDS = [71, 89]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory):
    values = {}
    for path in directory.glob("*/result_*.json"):
        result = json.loads(path.read_text())
        values[(result["arm"], result["seed"])] = result
    return values


def paired_bootstrap(left, right, depth, samples=20_000):
    left_docs = left["evaluation"]["per_document"][str(depth)]
    right_docs = right["evaluation"]["per_document"][str(depth)]
    documents = sorted(left_docs)
    if documents != sorted(right_docs):
        raise RuntimeError("Evaluation documents differ")
    left_sum = np.array([left_docs[d]["nll_sum"] for d in documents])
    right_sum = np.array([right_docs[d]["nll_sum"] for d in documents])
    tokens = np.array([left_docs[d]["tokens"] for d in documents])
    rng = np.random.default_rng(20260823 + left["seed"] * 100 + depth)
    values = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(documents), (stop - start, len(documents)))
        values[start:stop] = (left_sum[indices].sum(1) - right_sum[indices].sum(1)) / tokens[indices].sum(1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "aux_minus_final_nll": float((left_sum.sum() - right_sum.sum()) / tokens.sum()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_aux_better": float(np.mean(values < 0)),
        "documents": len(documents),
        "bootstrap_samples": samples,
    }


def main():
    results = {**load(E006), **load(E007)}
    required = [(arm, seed) for arm in (FINAL, AUX) for seed in SEEDS]
    missing = [key for key in required if key not in results]
    if missing:
        raise RuntimeError(f"Missing results: {missing}")

    rows = []
    for arm, seed in required:
        result = results[(arm, seed)]
        metrics = result["evaluation"]["metrics"]
        best = min(DEPTHS, key=lambda depth: metrics[str(depth)]["nll"])
        for depth in DEPTHS:
            rows.append({
                "arm": arm, "seed": seed, "depth": depth,
                "nll": metrics[str(depth)]["nll"], "ppl": metrics[str(depth)]["ppl"],
                "best_depth": best, "presented_tokens": result["presented_tokens"],
            })
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "E007_depth_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    gains = {depth: {} for depth in (16, 24)}
    best_depths, boundary_pass = {}, {}
    for seed in SEEDS:
        for depth in (16, 24):
            final_nll = results[(FINAL, seed)]["evaluation"]["metrics"][str(depth)]["nll"]
            aux_nll = results[(AUX, seed)]["evaluation"]["metrics"][str(depth)]["nll"]
            gains[depth][seed] = final_nll - aux_nll
        for arm in (FINAL, AUX):
            metrics = results[(arm, seed)]["evaluation"]["metrics"]
            best_depths[f"{arm}_s{seed}"] = min(DEPTHS, key=lambda depth: metrics[str(depth)]["nll"])
            boundary_pass[f"{arm}_s{seed}"] = (
                metrics["24"]["nll"] - metrics["16"]["nll"] <= 0.02
            )

    new_depth_pass = {
        str(depth): all(gains[depth][seed] >= 0.01 for seed in NEW_SEEDS) for depth in (16, 24)
    }
    aggregate = {}
    for depth in (16, 24):
        values = list(gains[depth].values())
        aggregate[str(depth)] = {
            "wins_at_least_0.01": sum(value >= 0.01 for value in values),
            "mean_final_minus_aux_nll": float(np.mean(values)),
            "passes": sum(value >= 0.01 for value in values) >= 3 and float(np.mean(values)) > 0,
        }
    decisions = {
        "R1_auxiliary_replication": any(new_depth_pass.values()),
        "R1_depth_thresholds": new_depth_pass,
        "R2_extended_range_stability": all(
            best_depths[f"{arm}_s{seed}"] in (16, 24) and boundary_pass[f"{arm}_s{seed}"]
            for arm in (FINAL, AUX) for seed in NEW_SEEDS
        ),
        "R3_aggregate_auxiliary_support": any(value["passes"] for value in aggregate.values()),
    }
    bootstrap = {
        str(seed): {
            str(depth): paired_bootstrap(results[(AUX, seed)], results[(FINAL, seed)], depth)
            for depth in (16, 24)
        } for seed in NEW_SEEDS
    }
    summary = {
        "seeds": SEEDS,
        "new_seeds": NEW_SEEDS,
        "final_minus_aux_nll": {str(depth): values for depth, values in gains.items()},
        "best_depths": best_depths,
        "boundary_pass": boundary_pass,
        "aggregate": aggregate,
        "paired_document_bootstrap_new_seeds": bootstrap,
        "registered_decision": decisions,
    }
    (REPORT / "E007_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    manifest = []
    for result_path in sorted(E007.glob("*/result_*.json")):
        result = json.loads(result_path.read_text())
        registration = result_path.parent / "registration.json"
        manifest.append({
            "arm": result["arm"], "seed": result["seed"], "parameters": result["parameters"],
            "presented_tokens": result["presented_tokens"], "training_seconds": result["training_seconds"],
            "peak_allocated_bytes": result["peak_allocated_bytes"], "skipped_updates": result["skipped_updates"],
            "checkpoint_sha256": result["checkpoint_sha256"], "result_sha256": sha256(result_path),
            "registration_sha256": sha256(registration),
        })
    (REPORT / "E007_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)
    for axis, seed in zip(axes.flat, SEEDS):
        for arm, label in ((FINAL, "final-only"), (AUX, "auxiliary")):
            metrics = results[(arm, seed)]["evaluation"]["metrics"]
            axis.plot(DEPTHS, [metrics[str(d)]["nll"] for d in DEPTHS], marker="o", label=label)
        axis.axvline(24, color="0.35", linestyle=":", linewidth=1)
        axis.set_xscale("log", base=2); axis.set_xticks(DEPTHS)
        axis.set(title=f"seed {seed}", xlabel="Inference depth T", ylabel="Validation NLL")
        axis.grid(alpha=0.25); axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "E007_seed_replication.png", dpi=180)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
