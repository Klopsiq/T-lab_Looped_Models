"""E002: registered GPU screening on the saved FineWeb byte snapshot."""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_models.data import load_windows, sha256
from looped_models.model import LoopedLM, ModelConfig


ARMS = {
    "relative_fixed16": dict(relative=True, depth_rope=False, variable=False, auxiliary=False),
    "combined_fixed16": dict(relative=True, depth_rope=True, variable=False, auxiliary=False),
    "combined_random8_16": dict(relative=True, depth_rope=True, variable=True, auxiliary=False),
    "combined_random8_16_aux": dict(relative=True, depth_rope=True, variable=True, auxiliary=True),
}
EVAL_DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def learning_rate(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - 1 - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.inference_mode()
def evaluate(model, windows, ids, depths, device, batch_size=16):
    model.eval()
    totals = {depth: 0.0 for depth in depths}
    per_document = {depth: {} for depth in depths}
    tokens = 0
    start = time.perf_counter()
    for offset in range(0, len(windows), batch_size):
        batch = torch.as_tensor(windows[offset:offset + batch_size].astype(np.int64), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
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
                record = per_document[depth].setdefault(doc, {"nll_sum": 0.0, "tokens": 0})
                record["nll_sum"] += row_losses.double().sum().item()
                record["tokens"] += row_losses.numel()
        tokens += targets.numel()
    torch.cuda.synchronize(device)
    return {
        "tokens": tokens,
        "documents": len(set(map(str, ids))),
        "metrics": {str(d): {"nll": totals[d] / tokens, "ppl": math.exp(totals[d] / tokens)} for d in depths},
        "per_document": {str(k): v for k, v in per_document.items()},
        "seconds": time.perf_counter() - start,
    }


def train_arm(args, arm_name, train, validation, validation_ids, registration):
    arm = ARMS[arm_name]
    directory = Path(args.output) / f"{arm_name}_s{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path = directory / "result.json", directory / "last.pt"
    if result_path.exists():
        print(f"{arm_name}: already complete", flush=True)
        return json.loads(result_path.read_text())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = ModelConfig(vocab_size=256, width=args.width, intermediate=args.intermediate,
                      heads=8, kv_heads=2, core_layers=2,
                      relative=arm["relative"], depth_rope=arm["depth_rope"])
    model = LoopedLM(cfg).to(args.device)
    if model.parameter_count() > 10_000_000:
        raise ValueError("Parameter budget exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    data_rng = torch.Generator().manual_seed(10_000 + args.seed)
    schedule_rng = torch.Generator().manual_seed(20_000 + args.seed)
    steps = args.tokens // (args.batch * args.context)
    warmup = max(10, min(50, steps // 10))
    start_step, prior_seconds, skipped = 0, 0.0, 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        data_rng.set_state(ckpt["data_rng"].cpu())
        schedule_rng.set_state(ckpt["schedule_rng"].cpu())
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        start_step, prior_seconds, skipped = ckpt["step"], ckpt["training_seconds"], ckpt["skipped_updates"]

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    with (directory / "learning_curve.jsonl").open("a") as log:
        for step in range(start_step, steps):
            model.train()
            lr = learning_rate(step, steps, args.lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            indices = torch.randint(len(train), (args.batch,), generator=data_rng)
            batch = torch.as_tensor(train[indices.numpy()].astype(np.int64), device=args.device)
            loops = int(torch.randint(8, 17, (1,), generator=schedule_rng)) if arm["variable"] else 16
            aux_weight = 0.3 * (1 - step / max(1, steps - 1)) if arm["auxiliary"] else 0.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if arm["auxiliary"]:
                    final_logits, states = model(batch[:, :-1], loops, collect=True, grad_checkpoint=True)
                    auxiliary_logits = model.readout(states[max(1, loops // 2)])
                else:
                    final_logits = model(batch[:, :-1], loops, grad_checkpoint=True)
                targets = batch[:, 1:]
                final_loss = torch.nn.functional.cross_entropy(final_logits.float().reshape(-1, 256), targets.reshape(-1))
                if arm["auxiliary"]:
                    auxiliary_loss = torch.nn.functional.cross_entropy(auxiliary_logits.float().reshape(-1, 256), targets.reshape(-1))
                    loss = (1 - aux_weight) * final_loss + aux_weight * auxiliary_loss
                else:
                    loss = final_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)

            should_save = (step + 1) % args.checkpoint_every == 0 or step + 1 == steps
            if should_save:
                torch.cuda.synchronize(args.device)
                elapsed = prior_seconds + time.perf_counter() - started
                checkpoint = {
                    "model": model.state_dict(), "model_config": model.config_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "data_rng": data_rng.get_state(), "schedule_rng": schedule_rng.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "step": step + 1, "training_seconds": elapsed, "skipped_updates": skipped,
                    "presented_tokens": (step + 1) * args.batch * args.context,
                    "arm": arm_name, "seed": args.seed, "registration_sha256": registration,
                }
                save_checkpoint(checkpoint_path, checkpoint)
                row = {"step": step + 1, "presented_tokens": checkpoint["presented_tokens"],
                       "train_nll": final_loss.detach().item(), "objective": loss.detach().item(), "loops": loops,
                       "aux_weight": aux_weight, "lr": lr, "gradient_norm": grad_norm.detach().item(),
                       "skipped_updates": skipped, "training_seconds": elapsed}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(arm_name, row, flush=True)

    final_eval = evaluate(model, validation, validation_ids, EVAL_DEPTHS, args.device)
    elapsed = prior_seconds + time.perf_counter() - started
    result = {"experiment": "E002", "arm": arm_name, "seed": args.seed,
              "parameters": model.parameter_count(), "presented_tokens": steps * args.batch * args.context,
              "training_seconds": elapsed, "skipped_updates": skipped,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
              "checkpoint_sha256": sha256(checkpoint_path.read_bytes()), "evaluation": final_eval}
    write_json(result_path, result)
    print(f"{arm_name}: complete; NLL@16={final_eval['metrics']['16']['nll']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--data", default="data/fineweb_pilot_v1")
    parser.add_argument("--output", default="runs/E002")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=5_000_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=244)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("E002 requires CUDA")
    if args.tokens > 100_000_000 or args.tokens < args.batch * args.context:
        raise ValueError("Invalid token budget")
    train, _ = load_windows(args.data, "train")
    validation, validation_ids = load_windows(args.data, "validation", 64)
    if train.shape[1] != args.context + 1:
        raise ValueError("Context mismatch")
    root = Path(__file__).resolve().parents[1]
    registration_path = root / "research/protocols/E002_gpu_screen.md"
    source_paths = [Path(__file__).resolve(), root / "looped_models/model.py", root / "looped_models/data.py",
                    root / args.data / "manifest.json", registration_path]
    registration = {str(p.relative_to(root)): sha256(p.read_bytes()) for p in source_paths}
    registration_sha = hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest()
    write_json(Path(args.output) / f"registration_{args.device.replace(':', '_')}.json",
               {"created_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "source_hashes": registration, "registration_sha256": registration_sha,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device)})
    for arm in args.arms:
        train_arm(args, arm, train, validation, validation_ids, registration_sha)


if __name__ == "__main__":
    main()
