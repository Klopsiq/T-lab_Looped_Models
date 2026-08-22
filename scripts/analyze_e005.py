"""Analyze E005 MLP-first transfers against the matched E003 controls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E005_rtx4080_20260914"
E003 = ROOT / "reports" / "E003_bpe" / "raw" / "runs" / "E003"
E005 = REPORT / "raw" / "runs" / "E005"
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
PAIRS = {
    "fixed": ("relative_fixed16", "relative_fixed16_mlp_first"),
    "auxiliary": ("combined_random8_16_aux", "combined_random8_16_aux_mlp_first"),
}


def load(directory):
    results = {}
    for path in directory.glob("*/result_*.json"):
        value = json.loads(path.read_text())
        key = (value["arm"], value["seed"])
        if key not in results or value["presented_tokens"] > results[key]["presented_tokens"]:
            results[key] = value
    return results


def paired_bootstrap(mlp_first, standard, depth, samples=20_000):
    left = mlp_first["evaluation"]["per_document"][str(depth)]
    right = standard["evaluation"]["per_document"][str(depth)]
    documents = sorted(left)
    if documents != sorted(right):
        raise RuntimeError("Evaluation documents differ")
    left_sum = np.array([left[d]["nll_sum"] for d in documents])
    right_sum = np.array([right[d]["nll_sum"] for d in documents])
    tokens = np.array([left[d]["tokens"] for d in documents])
    rng = np.random.default_rng(20260914 + mlp_first["seed"] * 100 + depth)
    values = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(documents), (stop - start, len(documents)))
        values[start:stop] = (left_sum[indices].sum(1) - right_sum[indices].sum(1)) / tokens[indices].sum(1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mlp_first_minus_standard_nll": float((left_sum.sum() - right_sum.sum()) / tokens.sum()),
        "ci95_low": float(low), "ci95_high": float(high),
        "probability_mlp_first_better": float(np.mean(values < 0)),
        "documents": len(documents), "bootstrap_samples": samples,
    }


def main():
    results = {**load(E003), **load(E005)}
    required = [(arm, seed) for pair in PAIRS.values() for arm in pair for seed in (23, 47)]
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
    with (REPORT / "E005_depth_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    comparisons = {}
    gains = {recipe: {depth: [] for depth in (12, 16)} for recipe in PAIRS}
    interactions = []
    for seed in (23, 47):
        for recipe, (standard_arm, mlp_arm) in PAIRS.items():
            standard, mlp = results[(standard_arm, seed)], results[(mlp_arm, seed)]
            comparisons[f"{recipe}_s{seed}"] = {
                str(depth): paired_bootstrap(mlp, standard, depth) for depth in (12, 16, 32)
            }
            for depth in (12, 16):
                gains[recipe][depth].append(
                    standard["evaluation"]["metrics"][str(depth)]["nll"]
                    - mlp["evaluation"]["metrics"][str(depth)]["nll"]
                )
        fixed_effect = -comparisons[f"fixed_s{seed}"]["16"]["mlp_first_minus_standard_nll"]
        aux_effect = -comparisons[f"auxiliary_s{seed}"]["16"]["mlp_first_minus_standard_nll"]
        interactions.append({"seed": seed, "nll_difference_of_differences": fixed_effect - aux_effect})

    fixed_h1 = all(value >= 0.02 for value in gains["fixed"][16])
    aux_depth_pass = {
        str(depth): all(value >= 0.02 for value in gains["auxiliary"][depth]) for depth in (12, 16)
    }
    aux_delta_pass = []
    for seed in (23, 47):
        standard = results[(PAIRS["auxiliary"][0], seed)]["evaluation"]["metrics"]
        mlp = results[(PAIRS["auxiliary"][1], seed)]["evaluation"]["metrics"]
        aux_delta_pass.append(mlp["32"]["nll"] - mlp["16"]["nll"] <= standard["32"]["nll"] - standard["16"]["nll"] + 0.02)
    h2 = any(aux_depth_pass.values()) and all(aux_delta_pass)
    summary = {
        "pairs": PAIRS, "seeds": [23, 47], "paired_document_bootstrap": comparisons,
        "quality_gain_standard_minus_mlp_first": {
            recipe: {str(depth): values for depth, values in depths.items()} for recipe, depths in gains.items()
        },
        "interaction_at_16": interactions,
        "mean_interaction_at_16": float(np.mean([row["nll_difference_of_differences"] for row in interactions])),
        "registered_decision": {"H1_fixed_transfer": fixed_h1, "H2_auxiliary_compatibility": h2,
                                "auxiliary_depth_thresholds": aux_depth_pass, "auxiliary_delta_pass": aux_delta_pass},
    }
    (REPORT / "E005_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    labels = {
        "relative_fixed16": "fixed, Attn→MLP", "relative_fixed16_mlp_first": "fixed, MLP→Attn",
        "combined_random8_16_aux": "aux, Attn→MLP",
        "combined_random8_16_aux_mlp_first": "aux, MLP→Attn",
    }
    for axis, seed in zip(axes, (23, 47)):
        for arm in labels:
            metrics = results[(arm, seed)]["evaluation"]["metrics"]
            axis.plot(DEPTHS, [metrics[str(d)]["nll"] for d in DEPTHS], marker="o", label=labels[arm])
        axis.axvline(16, color="0.4", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2); axis.set_xticks(DEPTHS)
        axis.set(title=f"seed {seed}", xlabel="Inference depth T", ylabel="Validation NLL")
        axis.grid(alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "E005_order_interaction.png", dpi=180)
    print(json.dumps(summary["registered_decision"], indent=2))


if __name__ == "__main__":
    main()
