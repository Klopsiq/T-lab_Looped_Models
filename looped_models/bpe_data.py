"""Deterministic loaders for packed uint16 BPE corpora."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import sha256


def load_manifest(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        if sha256((root / name).read_bytes()) != record["sha256"]:
            raise ValueError(f"Corpus hash mismatch: {name}")
    return manifest


class PackedTrain:
    def __init__(self, path: str | Path, context: int, seed: int):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.context = context
        self.starts = np.arange(0, len(self.tokens) - context - 1, context, dtype=np.int64)
        np.random.default_rng(seed).shuffle(self.starts)
        self.cursor = 0

    def batch(self, size: int, device: str) -> torch.Tensor:
        if self.cursor + size > len(self.starts):
            raise RuntimeError("The no-repeat training corpus is exhausted")
        starts = self.starts[self.cursor : self.cursor + size]
        self.cursor += size
        offsets = starts[:, None] + np.arange(self.context + 1)[None, :]
        rows = np.asarray(self.tokens[offsets], dtype=np.int64)
        return torch.from_numpy(rows).to(device, non_blocking=True)


def load_validation(root: str | Path):
    with np.load(Path(root) / "validation_windows.npz", allow_pickle=False) as data:
        return data["windows"].copy(), data["doc_ids"].copy()
