"""Analyze paired E003 results and apply the registered continuation rule."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E003_bpe"
RAW = REPORT / "raw" / "runs" / "E003"
BASELINE = "relative_fixed16"
CANDIDATE = "combined_random8_16_aux"
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]


def load_results():
    found = {}
    for path in RAW.glob("*/result_*.json"):
        result = json.loads(path.read_text())
        key = (result["arm"], result["seed"])
        if key not in found or result["presented_tokens"] > found[key]["presented_tokens"]:
            found[key] = result
    return found


def document_difference(candidate, baseline, depth, samples=20_000, seed=20260913):
    c_docs = candidate["evaluation"]["per_document"][str(depth)]
    b_docs = baseline["evaluation"]["per_document"][str(depth)]
    documents = sorted(c_docs)
    if documents != sorted(b_docs):
        raise RuntimeError("Paired evaluation documents differ")
    c_sum = np.array([c_docs[d]["nll_sum"] for d in documents])
    b_sum = np.array([b_docs[d]["nll_sum"] for d in documents])
    tokens = np.array([c_docs[d]["tokens"] for d in documents])
    rng = np.random.default_rng(seed + candidate["seed"] * 100 + depth)
    differences = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(documents), (stop - start, len(documents)))
        differences[start:stop] = (
            c_sum[indices].sum(1) - b_sum[indices].sum(1)
        ) / tokens[indices].sum(1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference_nll": float((c_sum.sum() - b_sum.sum()) / tokens.sum()),
        "ci95_low": float(low), "ci95_high": float(high),
        "probability_candidate_better": float(np.mean(differences < 0)),
        "documents": len(documents), "bootstrap_samples": samples,
    }


def main():
    results = load_results()
    seeds = sorted(seed for arm, seed in results if arm == BASELINE and (CANDIDATE, seed) in results)
    if not seeds:
        raise RuntimeError("No complete paired E003 seeds")
    rows, comparisons = [], {}
    for (arm, seed), result in sorted(results.items()):
        metrics = result["evaluation"]["metrics"]
        best = min(DEPTHS, key=lambda depth: metrics[str(depth)]["nll"])
        for depth in DEPTHS:
            rows.append({
                "arm": arm, "seed": seed, "presented_tokens": result["presented_tokens"], "depth": depth,
                "nll": metrics[str(depth)]["nll"], "ppl": metrics[str(depth)]["ppl"], "best_depth": best,
                "training_seconds": result["training_seconds"], "peak_allocated_bytes": result["peak_allocated_bytes"],
            })
    with (REPORT / "E003_depth_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    decisions = []
    for seed in seeds:
        baseline, candidate = results[(BASELINE, seed)], results[(CANDIDATE, seed)]
        comparisons[str(seed)] = {
            str(depth): document_difference(candidate, baseline, depth) for depth in DEPTHS
        }
        bm, cm = baseline["evaluation"]["metrics"], candidate["evaluation"]["metrics"]
        quality_gain = bm["16"]["nll"] - cm["16"]["nll"]
        baseline_delta = bm["32"]["nll"] - bm["16"]["nll"]
        candidate_delta = cm["32"]["nll"] - cm["16"]["nll"]
        delta_gain = baseline_delta - candidate_delta
        condition_1 = quality_gain >= 0.02 and candidate_delta <= baseline_delta
        condition_2 = delta_gain >= 0.02 and quality_gain >= -0.02
        decisions.append({
            "seed": seed, "candidate_quality_gain_nll_at_16": quality_gain,
            "baseline_delta_16_to_32": baseline_delta, "candidate_delta_16_to_32": candidate_delta,
            "candidate_delta_gain": delta_gain, "condition_1": condition_1, "condition_2": condition_2,
            "seed_passes": condition_1 or condition_2,
        })
    summary = {
        "baseline": BASELINE, "candidate": CANDIDATE, "seeds": seeds,
        "paired_document_bootstrap": comparisons, "registered_decision_by_seed": decisions,
        "continue_both_to_100m": len(seeds) == 2 and all(item["seed_passes"] for item in decisions),
    }
    (REPORT / "E003_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(seeds), figsize=(6.4 * len(seeds), 4.6), squeeze=False)
    for axis, seed in zip(axes[0], seeds):
        for arm, label in ((BASELINE, "relative fixed T=16"), (CANDIDATE, "variable T + auxiliary")):
            metrics = results[(arm, seed)]["evaluation"]["metrics"]
            axis.plot(DEPTHS, [metrics[str(d)]["nll"] for d in DEPTHS], marker="o", label=label)
        axis.axvline(16, color="0.4", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2); axis.set_xticks(DEPTHS)
        axis.set(title=f"seed {seed}", xlabel="Inference depth T", ylabel="Validation NLL")
        axis.grid(alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/E003_depth_curves.png", dpi=180)
    print(json.dumps(summary["registered_decision_by_seed"], indent=2))
    print("continue_both_to_100m=", summary["continue_both_to_100m"])


if __name__ == "__main__":
    main()
