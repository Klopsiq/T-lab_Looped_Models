"""Standalone evaluation of an E003 checkpoint on registered token windows."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_models.data import sha256
from looped_models.model import LoopedLM, ModelConfig


@torch.inference_mode()
def evaluate(model, windows, ids, depths, device, batch_size):
    totals = {depth: 0.0 for depth in depths}
    documents = {depth: {} for depth in depths}
    model.eval(); started = time.perf_counter(); tokens = 0
    for offset in range(0, len(windows), batch_size):
        batch = torch.as_tensor(windows[offset : offset + batch_size].astype(np.int64), device=device)
        amp = torch.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else contextlib.nullcontext()
        with amp:
            _, states = model(batch[:, :-1], max(depths), collect=True)
        targets = batch[:, 1:]
        for depth in depths:
            logits = model.readout(states[depth]).float()
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size), targets.reshape(-1), reduction="none"
            ).reshape_as(targets)
            totals[depth] += losses.double().sum().item()
            for row, row_losses in enumerate(losses):
                doc = str(ids[offset + row])
                record = documents[depth].setdefault(doc, {"nll_sum": 0.0, "tokens": 0})
                record["nll_sum"] += row_losses.double().sum().item(); record["tokens"] += row_losses.numel()
        tokens += targets.numel()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {
        "tokens": tokens, "documents": len(set(map(str, ids))),
        "metrics": {str(d): {"nll": totals[d] / tokens, "ppl": math.exp(totals[d] / tokens)} for d in depths},
        "per_document": {str(d): value for d, value in documents.items()},
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path, windows_path = Path(args.checkpoint), Path(args.windows)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = LoopedLM(ModelConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model"])
    with np.load(windows_path, allow_pickle=False) as data:
        windows, ids = data["windows"].copy(), data["doc_ids"].copy()
    result = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path.read_bytes()),
        "windows": str(windows_path), "windows_sha256": sha256(windows_path.read_bytes()),
        "depths": args.depths, "evaluation": evaluate(model, windows, ids, args.depths, args.device, args.batch),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["evaluation"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
