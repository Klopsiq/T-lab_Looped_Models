"""Build a leakage-aware 8k byte-level BPE corpus from one pinned FineWeb parquet."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EOS = "<|endoftext|>"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def split_for(group: str) -> str:
    bucket = int(group[:16], 16) % 100
    return "train" if bucket < 90 else ("validation" if bucket < 95 else "test")


def rows(path: Path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema.names)
    id_column = "id" if "id" in columns else None
    requested = ["text"] + ([id_column] if id_column else [])
    ordinal = 0
    for batch in parquet.iter_batches(columns=requested, batch_size=1024):
        values = batch.to_pydict()
        for index, text in enumerate(values["text"]):
            if text:
                identifier = values[id_column][index] if id_column else f"row:{ordinal}"
                yield str(identifier), str(text)
            ordinal += 1


def train_tokenizer(source: Path, output: Path, vocab_size: int, documents: int) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def training_texts():
        seen = set()
        accepted = 0
        for _, text in rows(source):
            group = normalized_hash(text)
            if group in seen or split_for(group) != "train":
                continue
            seen.add(group)
            yield text
            accepted += 1
            if accepted >= documents:
                return

    tokenizer.train_from_iterator(training_texts(), trainer=trainer, length=documents)
    if tokenizer.get_vocab_size() != vocab_size or tokenizer.token_to_id(EOS) != 0:
        raise RuntimeError("Tokenizer did not produce the registered vocabulary")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer.get_vocab_size()


def build_corpus(args, tokenizer_path: Path) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id(EOS)
    caps = {"train": args.train_tokens, "validation": args.validation_tokens, "test": args.test_tokens}
    counts = {split: {"tokens": 0, "utf8_bytes": 0, "documents": 0} for split in caps}
    seen = set()
    duplicates = 0
    temporary = {split: Path(args.output) / f"{split}.bin.tmp" for split in caps}
    handles = {split: path.open("wb") for split, path in temporary.items()}
    eval_candidates = []
    try:
        for identifier, text in rows(Path(args.source)):
            group = normalized_hash(text)
            if group in seen:
                duplicates += 1
                continue
            seen.add(group)
            split = split_for(group)
            if counts[split]["tokens"] >= caps[split]:
                if all(counts[name]["tokens"] >= caps[name] for name in caps):
                    break
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
            array = np.asarray(ids, dtype="<u2")
            array.tofile(handles[split])
            counts[split]["tokens"] += len(ids)
            counts[split]["utf8_bytes"] += len(text.encode("utf-8"))
            counts[split]["documents"] += 1
            if split == "validation":
                for start in range(0, len(ids) - args.context, args.context):
                    window = array[start : start + args.context + 1].copy()
                    score = int(hashlib.sha256(f"{group}:{start}".encode()).hexdigest()[:16], 16)
                    eval_candidates.append((score, group, window))
    finally:
        for handle in handles.values():
            handle.close()

    if any(counts[split]["tokens"] < caps[split] for split in caps):
        raise RuntimeError(f"Source ended before corpus targets were reached: {counts}")
    for split, path in temporary.items():
        path.replace(Path(args.output) / f"{split}.bin")

    selected = heapq.nsmallest(args.eval_windows, eval_candidates, key=lambda item: item[0])
    if len(selected) < args.eval_windows:
        raise RuntimeError("Not enough within-document validation windows")
    eval_path = Path(args.output) / "validation_windows.npz"
    np.savez_compressed(
        eval_path,
        windows=np.stack([item[2] for item in selected]),
        doc_ids=np.asarray([item[1] for item in selected]),
    )
    return {
        "counts": counts,
        "duplicates_removed": duplicates,
        "unique_documents_seen": len(seen),
        "evaluation": {
            "file": eval_path.name,
            "windows": len(selected),
            "documents": len({item[1] for item in selected}),
            "target_tokens": len(selected) * args.context,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", default="data/fineweb_bpe8k_v1")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-documents", type=int, default=50_000)
    parser.add_argument("--train-tokens", type=int, default=110_000_000)
    parser.add_argument("--validation-tokens", type=int, default=4_000_000)
    parser.add_argument("--test-tokens", type=int, default=4_000_000)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--eval-windows", type=int, default=2048)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Corpus already prepared; choose a new output directory")
    actual_source_sha = file_sha256(source)
    if actual_source_sha != args.source_sha256:
        raise ValueError(f"Source SHA-256 mismatch: {actual_source_sha}")

    tokenizer_path = output / "tokenizer.json"
    vocabulary = train_tokenizer(source, tokenizer_path, args.vocab_size, args.tokenizer_documents)
    corpus = build_corpus(args, tokenizer_path)
    files = {}
    for name in ("train.bin", "validation.bin", "test.bin", "validation_windows.npz", "tokenizer.json"):
        path = output / name
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "source": {"url": args.source_url, "revision": args.source_revision, "sha256": actual_source_sha},
        "split": "sha256(whitespace-normalized full text) mod 100: 0..89 train, 90..94 validation, 95..99 test",
        "dedup": "exact whitespace-normalized text before split",
        "tokenizer": {
            "type": "byte-level BPE",
            "vocab_size": vocabulary,
            "training_documents": args.tokenizer_documents,
            "training_split": "train only",
            "eos_token": EOS,
            "eos_id": 0,
        },
        "packing": "uint16 document streams separated by EOS; training windows may cross EOS",
        "context": args.context,
        **corpus,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
