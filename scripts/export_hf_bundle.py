"""Export the selected custom LoopedLM checkpoint as a self-contained HF bundle."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_models.data import sha256


INFERENCE = '''"""Minimal greedy generation example for the custom LoopedLM checkpoint."""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import LoopedLM, ModelConfig

root = Path(__file__).resolve().parent
artifact = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
model = LoopedLM(ModelConfig(**artifact["model_config"]))
model.load_state_dict(artifact["model"]); model.eval()
tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

def generate(prompt: str, max_new_tokens: int = 50, loops: int = 12) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            tokens = torch.tensor([ids[-512:]], dtype=torch.long)
            logits = model(tokens, loops=loops)
            ids.append(int(logits[0, -1].argmax()))
    return tokenizer.decode(ids)

if __name__ == "__main__":
    print(generate("Looped transformers", max_new_tokens=40))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-result", required=True)
    parser.add_argument("--test-result", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", default="artifacts/looped-models-bpe8k-final")
    parser.add_argument("--inference-depth", type=int, default=12)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_hash = sha256(checkpoint_path.read_bytes())
    validation = json.loads(Path(args.validation_result).read_text())
    test = json.loads(Path(args.test_result).read_text())
    if validation["checkpoint_sha256"] != checkpoint_hash or test["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Selected result and checkpoint hashes do not match")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": checkpoint["model"], "model_config": checkpoint["model_config"],
        "inference_depth": args.inference_depth, "source_checkpoint_sha256": checkpoint_hash,
        "presented_tokens": checkpoint["presented_tokens"], "arm": checkpoint["arm"], "seed": checkpoint["seed"],
    }
    torch.save(artifact, output / "model.pt")
    config = {key: value for key, value in artifact.items() if key != "model"}
    config["parameter_count"] = sum(tensor.numel() for tensor in artifact["model"].values())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(args.tokenizer, output / "tokenizer.json")
    shutil.copyfile(args.data_manifest, output / "data_manifest.json")
    shutil.copyfile(args.validation_result, output / "validation_result.json")
    shutil.copyfile(args.test_result, output / "test_result.json")
    shutil.copyfile(Path(__file__).resolve().parents[1] / "looped_models/model.py", output / "model.py")
    (output / "inference_example.py").write_text(INFERENCE)
    (output / "requirements.txt").write_text("torch>=2.4\ntokenizers>=0.20,<1\n")
    ppl = test["evaluation"]["metrics"][str(args.inference_depth)]["ppl"]
    nll = test["evaluation"]["metrics"][str(args.inference_depth)]["nll"]
    (output / "README.md").write_text(f'''---
language: en
license: other
library_name: pytorch
tags:
- text-generation
- looped-transformer
- fineweb
---

# LoopedLM BPE8k, auxiliary anytime training

This is the checkpoint selected before a single locked-test evaluation in the Looped Models research project.

The training dataset is FineWeb (ODC-By). No separate license has yet been selected for the model artifact.

- Architecture: two shared Qwen-style blocks, width 512, GQA 8/2, relative input injection, Depth-RoPE.
- Parameters: 9,440,513 unique parameters with tied input/output embeddings.
- Training: 24,969,216 FineWeb BPE tokens, context 512, depth sampled uniformly from 8 to 16, decaying intermediate LM loss.
- Selected inference depth: T={args.inference_depth}.
- Locked test: NLL {nll:.4f}, PPL {ppl:.2f} on 1,048,576 tokens from 1,251 documents.

The method improved validation and test quality inside the training-depth range, but extrapolation beyond T=16 degraded faster than the fixed-depth baseline. This checkpoint does not demonstrate useful test-time scaling to arbitrarily many loops.

## Use

Install `requirements.txt` and run `python inference_example.py`. This repository uses custom PyTorch code rather than a Transformers `AutoModel` class. `model.pt` contains weights and the model configuration, without optimizer state. Exact tokenizer and data hashes are stored in `data_manifest.json`; measured results are included as JSON.

## Limitations

This is a 9.44M-parameter research model trained on one pinned FineWeb shard for about 25M tokens. It is not intended for factual use, instruction following, or deployment. The model card reports a single held-out test opening after all selection decisions.
''')
    manifest = {
        path.name: sha256(path.read_bytes())
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "files": manifest}, indent=2))


if __name__ == "__main__":
    main()
