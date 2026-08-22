"""E005: test MLP-first order in the two E003 training recipes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_models.bpe_data import PackedTrain, load_manifest, load_validation
from looped_models.data import sha256
from looped_models.model import LoopedLM, ModelConfig


ARMS = {
    "relative_fixed16_mlp_first": dict(relative=True, depth_rope=False, variable=False, auxiliary=False),
    "combined_random8_16_aux_mlp_first": dict(relative=True, depth_rope=True, variable=True, auxiliary=True),
}
DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def learning_rate(step, schedule_steps, peak, warmup_fraction=0.02):
    warmup = max(1, int(schedule_steps * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, schedule_steps - 1 - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.inference_mode()
def evaluate(model, windows, ids, device, batch_size):
    model.eval()
    totals = {depth: 0.0 for depth in DEPTHS}
    documents = {depth: {} for depth in DEPTHS}
    tokens = 0
    started = time.perf_counter()
    for offset in range(0, len(windows), batch_size):
        batch = torch.as_tensor(windows[offset : offset + batch_size].astype(np.int64), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            _, states = model(batch[:, :-1], max(DEPTHS), collect=True)
        targets = batch[:, 1:]
        for depth in DEPTHS:
            logits = model.readout(states[depth]).float()
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size), targets.reshape(-1), reduction="none"
            ).reshape_as(targets)
            totals[depth] += losses.double().sum().item()
            for row, row_losses in enumerate(losses):
                document = str(ids[offset + row])
                record = documents[depth].setdefault(document, {"nll_sum": 0.0, "tokens": 0})
                record["nll_sum"] += row_losses.double().sum().item()
                record["tokens"] += row_losses.numel()
        tokens += targets.numel()
    torch.cuda.synchronize(device)
    return {
        "tokens": tokens,
        "documents": len(set(map(str, ids))),
        "metrics": {
            str(depth): {"nll": totals[depth] / tokens, "ppl": math.exp(totals[depth] / tokens)}
            for depth in DEPTHS
        },
        "per_document": {str(depth): value for depth, value in documents.items()},
        "seconds": time.perf_counter() - started,
    }


def registration(args, arm, seed, manifest):
    root = Path(__file__).resolve().parents[1]
    run = Path(args.output) / f"{arm}_s{seed}"
    paths = [
        Path(__file__).resolve(), root / "looped_models/model.py", root / "looped_models/bpe_data.py",
        root / "research/protocols/E005_order_interaction.md",
        root / "research/protocols/E005_incident_01.md", root / args.data / "manifest.json",
        root / args.data / "tokenizer.json",
    ]
    hashes = {str(path.relative_to(root)): sha256(path.read_bytes()) for path in paths}
    registered_arguments = {key: value for key, value in vars(args).items() if key != "target_tokens"}
    value = {
        "created_at": datetime.now(timezone.utc).isoformat(), "arguments": registered_arguments, "arm": arm, "seed": seed,
        "source_hashes": hashes, "data_files": manifest["files"], "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
    }
    value["registration_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = run / "registration.json"
    if path.exists():
        old = json.loads(path.read_text())
        stable_keys = ("arguments", "arm", "seed", "source_hashes", "data_files", "torch", "cuda", "gpu")
        if any(old[key] != value[key] for key in stable_keys):
            raise RuntimeError(f"Registration changed for {run}")
        return old
    else:
        write_json(path, value)
        snapshot = run / "source_snapshot"
        for source in paths[:-2]:
            destination = snapshot / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return value


def train(args, arm_name, seed, manifest, validation, validation_ids):
    arm = ARMS[arm_name]
    directory = Path(args.output) / f"{arm_name}_s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    target_steps = args.target_tokens // (args.batch * args.context)
    schedule_steps = args.schedule_tokens // (args.batch * args.context)
    actual_target = target_steps * args.batch * args.context
    result_path = directory / f"result_{actual_target}.json"
    if result_path.exists():
        print(f"{arm_name} seed={seed}: target already complete", flush=True)
        return
    registered = registration(args, arm_name, seed, manifest)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    schedule_rng = torch.Generator().manual_seed(20_000 + seed)
    cfg = ModelConfig(vocab_size=manifest["tokenizer"]["vocab_size"], width=512, intermediate=1280,
                      heads=8, kv_heads=2, core_layers=2, relative=arm["relative"],
                      depth_rope=arm["depth_rope"], mlp_first=True)
    model = LoopedLM(cfg).to(args.device)
    if model.parameter_count() > 10_000_000:
        raise ValueError("Parameter budget exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    corpus = PackedTrain(Path(args.data) / "train.bin", args.context, 10_000 + seed)
    start_step, prior_seconds, skipped = 0, 0.0, 0
    checkpoint_path = directory / "last.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        schedule_rng.set_state(checkpoint["schedule_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        start_step, prior_seconds, skipped = checkpoint["step"], checkpoint["training_seconds"], checkpoint["skipped_updates"]
        corpus.cursor = checkpoint["data_cursor"]
    if start_step > target_steps:
        raise ValueError("Checkpoint is beyond requested target")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    with (directory / "learning_curve.jsonl").open("a") as log:
        for step in range(start_step, target_steps):
            model.train()
            lr = learning_rate(step, schedule_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = corpus.batch(args.batch, args.device)
            loops = int(torch.randint(8, 17, (1,), generator=schedule_rng)) if arm["variable"] else 16
            aux_weight = 0.3 * (1 - step / max(1, schedule_steps - 1)) if arm["auxiliary"] else 0.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if arm["auxiliary"]:
                    final_logits, states = model(batch[:, :-1], loops, collect=True, grad_checkpoint=True)
                    auxiliary_logits = model.readout(states[max(1, loops // 2)])
                else:
                    final_logits = model(batch[:, :-1], loops, grad_checkpoint=True)
                targets = batch[:, 1:]
                final_loss = torch.nn.functional.cross_entropy(
                    final_logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1)
                )
                if arm["auxiliary"]:
                    auxiliary_loss = torch.nn.functional.cross_entropy(
                        auxiliary_logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1)
                    )
                    loss = (1 - aux_weight) * final_loss + aux_weight * auxiliary_loss
                else:
                    loss = final_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            should_save = (step + 1) % args.checkpoint_every == 0 or step + 1 == target_steps
            if should_save:
                torch.cuda.synchronize(args.device)
                elapsed = prior_seconds + time.perf_counter() - started
                checkpoint = {
                    "model": model.state_dict(), "model_config": model.config_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "schedule_rng": schedule_rng.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "step": step + 1, "data_cursor": corpus.cursor,
                    "training_seconds": elapsed, "skipped_updates": skipped, "presented_tokens": (step + 1) * args.batch * args.context,
                    "arm": arm_name, "seed": seed, "registration_sha256": registered["registration_sha256"],
                }
                save_checkpoint(checkpoint_path, checkpoint)
                row = {
                    "step": step + 1, "presented_tokens": checkpoint["presented_tokens"],
                    "train_nll": final_loss.detach().item(), "objective": loss.detach().item(), "loops": loops,
                    "aux_weight": aux_weight, "lr": lr, "gradient_norm": gradient_norm.detach().item(),
                    "skipped_updates": skipped, "training_seconds": elapsed,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(arm_name, seed, row, flush=True)

    evaluation = evaluate(model, validation, validation_ids, args.device, args.eval_batch)
    elapsed = prior_seconds + time.perf_counter() - started
    result = {
        "experiment": "E005", "arm": arm_name, "seed": seed, "parameters": model.parameter_count(),
        "presented_tokens": actual_target, "schedule_tokens": args.schedule_tokens, "training_seconds": elapsed,
        "skipped_updates": skipped, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "checkpoint_sha256": sha256(checkpoint_path.read_bytes()), "evaluation": evaluation,
    }
    write_json(result_path, result)
    print(f"{arm_name} seed={seed}: complete NLL@16={evaluation['metrics']['16']['nll']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data", default="data/fineweb_bpe8k_v1")
    parser.add_argument("--output", default="runs/E005")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--schedule-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--checkpoint-every", type=int, default=192)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--eval-windows", type=int, default=2048)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("E005 requires CUDA")
    if args.target_tokens > args.schedule_tokens or args.schedule_tokens > 100_000_000:
        raise ValueError("Invalid token budget")
    manifest = load_manifest(args.data)
    if manifest["context"] != args.context or manifest["tokenizer"]["vocab_size"] != 8192:
        raise ValueError("E005 corpus does not match registered context/vocabulary")
    validation, validation_ids = load_validation(args.data)
    validation, validation_ids = validation[: args.eval_windows], validation_ids[: args.eval_windows]
    for seed in args.seeds:
        for arm in args.arms:
            train(args, arm, seed, manifest, validation, validation_ids)


if __name__ == "__main__":
    main()
