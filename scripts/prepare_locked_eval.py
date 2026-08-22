"""Reconstruct deterministic within-document windows for a registered corpus split."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_bpe_data import file_sha256, normalized_hash, rows, split_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", default="data/fineweb_bpe8k_v1")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--windows", type=int, default=2048)
    parser.add_argument("--exclude-windows", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    if file_sha256(Path(args.source)) != manifest["source"]["sha256"]:
        raise ValueError("Source parquet does not match registered corpus")
    if file_sha256(data / "tokenizer.json") != manifest["files"]["tokenizer.json"]["sha256"]:
        raise ValueError("Tokenizer does not match registered corpus")
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    eos = manifest["tokenizer"]["eos_id"]
    context = manifest["context"]
    caps = {name: manifest["counts"][name]["tokens"] for name in ("train", "validation", "test")}
    excluded_documents = set()
    exclusion_records = []
    for name in args.exclude_windows:
        path = Path(name)
        with np.load(path, allow_pickle=False) as prior:
            excluded_documents.update(map(str, prior["doc_ids"]))
        exclusion_records.append({"file": path.name, "sha256": file_sha256(path)})
    counts = {name: 0 for name in caps}
    seen, candidates = set(), []
    for _, text in rows(Path(args.source)):
        group = normalized_hash(text)
        if group in seen:
            continue
        seen.add(group)
        split = split_for(group)
        if counts[split] >= caps[split]:
            if all(counts[name] >= caps[name] for name in caps):
                break
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
        counts[split] += len(ids)
        if split == args.split and group not in excluded_documents:
            array = np.asarray(ids, dtype="<u2")
            for start in range(0, len(ids) - context, context):
                score = int(hashlib.sha256(f"{group}:{start}".encode()).hexdigest()[:16], 16)
                candidates.append((score, group, array[start : start + context + 1].copy()))
    selected = heapq.nsmallest(args.windows, candidates, key=lambda item: item[0])
    if len(selected) != args.windows or counts != caps:
        raise RuntimeError(
            f"Could not reconstruct registered traversal: counts={counts}, "
            f"eligible_windows={len(candidates)}, requested_windows={args.windows}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, windows=np.stack([x[2] for x in selected]),
                        doc_ids=np.asarray([x[1] for x in selected]))
    record = {
        "split": args.split, "windows": len(selected), "target_tokens": len(selected) * context,
        "documents": len({x[1] for x in selected}), "file": output.name,
        "sha256": file_sha256(output), "source_sha256": manifest["source"]["sha256"],
        "tokenizer_sha256": manifest["files"]["tokenizer.json"]["sha256"],
        "selection": "smallest sha256(document_hash:start), within-document non-overlapping windows",
        "excluded_documents": len(excluded_documents),
        "exclusion_files": exclusion_records,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
