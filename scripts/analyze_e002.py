"""Create reproducible tables, uncertainty estimates, and a figure for E002."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E002_gpu"
RAW = REPORT / "raw" / "runs" / "E002"
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
BASELINE = "relative_fixed16"


def load_results() -> dict[str, dict]:
    results = {}
    for path in sorted(RAW.glob("*/result.json")):
        result = json.loads(path.read_text())
        results[result["arm"]] = result
    if len(results) != 4:
        raise RuntimeError(f"Expected four E002 arms, found {len(results)}")
    return results


def paired_document_bootstrap(
    candidate: dict, baseline: dict, depth: int, *, samples: int = 20_000, seed: int = 20260913
) -> dict[str, float]:
    candidate_docs = candidate["evaluation"]["per_document"][str(depth)]
    baseline_docs = baseline["evaluation"]["per_document"][str(depth)]
    documents = sorted(candidate_docs)
    if documents != sorted(baseline_docs):
        raise RuntimeError("Candidate and baseline do not contain the same validation documents")

    candidate_sums = np.array([candidate_docs[d]["nll_sum"] for d in documents])
    baseline_sums = np.array([baseline_docs[d]["nll_sum"] for d in documents])
    tokens = np.array([candidate_docs[d]["tokens"] for d in documents])
    if not np.array_equal(tokens, [baseline_docs[d]["tokens"] for d in documents]):
        raise RuntimeError("Candidate and baseline document token counts differ")

    rng = np.random.default_rng(seed + depth)
    differences = np.empty(samples)
    for start in range(0, samples, 1_000):
        stop = min(samples, start + 1_000)
        indices = rng.integers(0, len(documents), size=(stop - start, len(documents)))
        denominators = tokens[indices].sum(axis=1)
        differences[start:stop] = (
            candidate_sums[indices].sum(axis=1) - baseline_sums[indices].sum(axis=1)
        ) / denominators
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference_nll": float((candidate_sums.sum() - baseline_sums.sum()) / tokens.sum()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_candidate_better": float(np.mean(differences < 0)),
        "documents": len(documents),
        "bootstrap_samples": samples,
    }


def write_outputs(results: dict[str, dict]) -> None:
    rows = []
    for arm, result in results.items():
        metrics = result["evaluation"]["metrics"]
        best_depth = min(DEPTHS, key=lambda depth: metrics[str(depth)]["nll"])
        for depth in DEPTHS:
            rows.append(
                {
                    "arm": arm,
                    "seed": result["seed"],
                    "depth": depth,
                    "nll": metrics[str(depth)]["nll"],
                    "ppl": metrics[str(depth)]["ppl"],
                    "best_depth": best_depth,
                    "training_seconds": result["training_seconds"],
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "presented_tokens": result["presented_tokens"],
                }
            )
    with (REPORT / "E002_depth_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    comparisons = {}
    for arm, result in results.items():
        if arm == BASELINE:
            continue
        comparisons[arm] = {
            str(depth): paired_document_bootstrap(result, results[BASELINE], depth)
            for depth in DEPTHS
        }
    (REPORT / "E002_paired_bootstrap.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "resampling_unit": "validation document",
                "interpretation": "negative difference favors candidate",
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )


def plot(results: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "relative_fixed16": "relative, fixed T=16",
        "combined_fixed16": "+ Depth-RoPE, fixed T=16",
        "combined_random8_16": "+ variable T=8…16",
        "combined_random8_16_aux": "+ variable T + auxiliary loss",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for arm in labels:
        metrics = results[arm]["evaluation"]["metrics"]
        ax.plot(DEPTHS, [metrics[str(d)]["nll"] for d in DEPTHS], marker="o", label=labels[arm])
    ax.axvline(16, color="0.4", linestyle="--", linewidth=1, label="maximum training depth")
    ax.set(xlabel="Inference depth T", ylabel="Validation NLL", title="E002: quality across recurrent depth")
    ax.set_xticks(DEPTHS)
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = ROOT / "reports" / "figures" / "E002_depth_curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    loaded = load_results()
    write_outputs(loaded)
    plot(loaded)
