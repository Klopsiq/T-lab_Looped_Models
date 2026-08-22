"""Apply the preregistered E008 decisions to the second locked test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "E008"
RAW = REPORT / "raw" / "runs" / "E008"
NEW = {seed: f"new_final_s{seed}.json" for seed in (23, 47, 71, 89)}
OLD = {seed: f"old_aux_s{seed}.json" for seed in (23, 47)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((RAW / name).read_text())


def paired_bootstrap(left, left_depth, right, right_depth, seed, samples=20_000):
    left_docs = left["evaluation"]["per_document"][str(left_depth)]
    right_docs = right["evaluation"]["per_document"][str(right_depth)]
    documents = sorted(left_docs)
    if documents != sorted(right_docs):
        raise RuntimeError("Test documents differ")
    left_sum = np.array([left_docs[d]["nll_sum"] for d in documents])
    right_sum = np.array([right_docs[d]["nll_sum"] for d in documents])
    tokens = np.array([left_docs[d]["tokens"] for d in documents])
    rng = np.random.default_rng(20260823 + seed * 100 + left_depth + right_depth)
    values = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(documents), (stop - start, len(documents)))
        values[start:stop] = (left_sum[indices].sum(1) - right_sum[indices].sum(1)) / tokens[indices].sum(1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "candidate_minus_previous_nll": float((left_sum.sum() - right_sum.sum()) / tokens.sum()),
        "ci95_low": float(low), "ci95_high": float(high),
        "probability_candidate_better": float(np.mean(values < 0)),
        "documents": len(documents), "bootstrap_samples": samples,
    }


def main():
    new = {seed: load(name) for seed, name in NEW.items()}
    old = {seed: load(name) for seed, name in OLD.items()}
    expected_windows = {value["windows_sha256"] for value in [*new.values(), *old.values()]}
    if len(expected_windows) != 1:
        raise RuntimeError("Evaluations used different test windows")

    q1, q2 = {}, {}
    for seed in (23, 47):
        q1[str(seed)] = paired_bootstrap(new[seed], 16, old[seed], 12, seed)
        q2[str(seed)] = paired_bootstrap(new[seed], 16, old[seed], 16, seed)
    q1_gains = [-q1[str(seed)]["candidate_minus_previous_nll"] for seed in (23, 47)]
    q3_deltas = {
        str(seed): new[seed]["evaluation"]["metrics"]["24"]["nll"]
        - new[seed]["evaluation"]["metrics"]["16"]["nll"] for seed in NEW
    }
    decisions = {
        "Q1_quality": all(value > 0 for value in q1_gains) and float(np.mean(q1_gains)) >= 0.01,
        "Q1_mean_nll_gain": float(np.mean(q1_gains)),
        "Q2_matched_compute": all(q2[str(seed)]["candidate_minus_previous_nll"] < 0 for seed in (23, 47)),
        "Q3_range_stability": all(delta <= 0.02 for delta in q3_deltas.values()),
        "Q3_nll_24_minus_16": q3_deltas,
    }
    metrics = {
        "candidate": {str(seed): new[seed]["evaluation"]["metrics"] for seed in NEW},
        "previous": {str(seed): old[seed]["evaluation"]["metrics"] for seed in OLD},
    }
    manifest = []
    for path in sorted(RAW.glob("*.json")):
        value = json.loads(path.read_text())
        manifest.append({
            "file": path.name, "sha256": sha256(path),
            "checkpoint_sha256": value["checkpoint_sha256"],
            "windows_sha256": value["windows_sha256"],
            "tokens": value["evaluation"]["tokens"],
            "documents": value["evaluation"]["documents"],
            "seconds": value["evaluation"]["seconds"],
        })
    result = {
        "windows_sha256": next(iter(expected_windows)), "metrics": metrics,
        "paired_bootstrap_Q1": q1, "paired_bootstrap_Q2": q2,
        "registered_decision": decisions, "run_manifest": manifest,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "E008_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
