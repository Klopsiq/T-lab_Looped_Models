"""Analyze E006 against its preregistered depth-range hypotheses."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E006"
E003 = ROOT / "reports" / "E003_bpe" / "raw" / "runs" / "E003"
E006 = REPORT / "raw" / "runs" / "E006"
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
OLD_AUX = "combined_random8_16_aux"
FINAL = "combined_random8_24_final"
AUX = "combined_random8_24_aux"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory):
    results = {}
    for path in directory.glob("*/result_*.json"):
        value = json.loads(path.read_text())
        key = (value["arm"], value["seed"])
        if key not in results or value["presented_tokens"] > results[key]["presented_tokens"]:
            results[key] = value
    return results


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
        "left_minus_right_nll": float((left_sum.sum() - right_sum.sum()) / tokens.sum()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_left_better": float(np.mean(values < 0)),
        "documents": len(documents),
        "bootstrap_samples": samples,
    }


def main():
    old, new = load(E003), load(E006)
    results = {**old, **new}
    required = [(arm, seed) for arm in (OLD_AUX, FINAL, AUX) for seed in (23, 47)]
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
                "arm": arm,
                "seed": seed,
                "depth": depth,
                "nll": metrics[str(depth)]["nll"],
                "ppl": metrics[str(depth)]["ppl"],
                "best_depth": best,
                "presented_tokens": result["presented_tokens"],
            })
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "E006_depth_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    bootstrap = {}
    decisions = {"per_seed": {}}
    h1_values, h3_values = [], []
    aux_gains = {16: [], 24: []}
    for seed in (23, 47):
        old_aux = results[(OLD_AUX, seed)]
        final = results[(FINAL, seed)]
        aux = results[(AUX, seed)]
        old_metrics = old_aux["evaluation"]["metrics"]
        final_metrics = final["evaluation"]["metrics"]
        aux_metrics = aux["evaluation"]["metrics"]
        old_16_24 = old_metrics["24"]["nll"] - old_metrics["16"]["nll"]
        new_16_24 = aux_metrics["24"]["nll"] - aux_metrics["16"]["nll"]
        new_24_32 = aux_metrics["32"]["nll"] - aux_metrics["24"]["nll"]
        boundary_reduction = old_16_24 - new_16_24
        h1 = new_16_24 <= 0.01 and boundary_reduction >= 0.03
        h3 = new_24_32 <= 0.02
        h1_values.append(h1)
        h3_values.append(h3)
        for depth in (16, 24):
            aux_gains[depth].append(final_metrics[str(depth)]["nll"] - aux_metrics[str(depth)]["nll"])
        decisions["per_seed"][str(seed)] = {
            "old_aux_nll_24_minus_16": old_16_24,
            "new_aux_nll_24_minus_16": new_16_24,
            "boundary_degradation_reduction": boundary_reduction,
            "new_aux_nll_32_minus_24": new_24_32,
            "H1": h1,
            "H3": h3,
        }
        bootstrap[str(seed)] = {
            "aux_minus_final": {
                str(depth): paired_bootstrap(aux, final, depth) for depth in (16, 24, 32)
            },
            "new_aux_minus_old_aux": {
                str(depth): paired_bootstrap(aux, old_aux, depth) for depth in (16, 24, 32)
            },
        }
    h2_depths = {str(depth): all(gain >= 0.01 for gain in values) for depth, values in aux_gains.items()}
    decisions.update({
        "H1_boundary_shift": all(h1_values),
        "H2_auxiliary_contribution": any(h2_depths.values()),
        "H2_depth_thresholds": h2_depths,
        "H3_beyond_range_extrapolation": all(h3_values),
    })
    summary = {
        "arms": {"old_control": OLD_AUX, "final_only": FINAL, "auxiliary": AUX},
        "seeds": [23, 47],
        "auxiliary_gain_final_minus_aux": {str(k): v for k, v in aux_gains.items()},
        "paired_document_bootstrap": bootstrap,
        "registered_decision": decisions,
    }
    (REPORT / "E006_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    run_manifest = []
    for result_path in sorted(E006.glob("*/result_*.json")):
        result = json.loads(result_path.read_text())
        registration_path = result_path.parent / "registration.json"
        run_manifest.append({
            "arm": result["arm"],
            "seed": result["seed"],
            "parameters": result["parameters"],
            "presented_tokens": result["presented_tokens"],
            "training_seconds": result["training_seconds"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "skipped_updates": result["skipped_updates"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "result_sha256": sha256(result_path),
            "registration_sha256": sha256(registration_path),
        })
    (REPORT / "E006_run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    labels = {OLD_AUX: "random 8–16, aux", FINAL: "random 8–24, final", AUX: "random 8–24, aux"}
    for axis, seed in zip(axes, (23, 47)):
        for arm, label in labels.items():
            metrics = results[(arm, seed)]["evaluation"]["metrics"]
            axis.plot(DEPTHS, [metrics[str(d)]["nll"] for d in DEPTHS], marker="o", label=label)
        axis.axvline(16, color="0.55", linestyle="--", linewidth=1)
        axis.axvline(24, color="0.25", linestyle=":", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(DEPTHS)
        axis.set(title=f"seed {seed}", xlabel="Inference depth T", ylabel="Validation NLL")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "E006_depth_range.png", dpi=180)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
