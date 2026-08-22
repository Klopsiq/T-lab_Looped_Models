"""Registered CPU/GPU pilot runner; also independently evaluate saved checkpoints."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_windows, sha256
from .model import LoopedLM, ModelConfig


ARMS = {
    'add': {}, 'relative': {'relative': True}, 'depth_rope': {'depth_rope': True},
    'combined': {'relative': True, 'depth_rope': True},
    'normalized_skip': {'relative': True, 'normalize_skip': True}, 'shallow': {},
}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def atomically_save(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


@torch.no_grad()
def evaluate(model, windows, ids, depths, device='cpu', batch_size=8):
    model.eval()
    totals = {t: 0.0 for t in depths}
    documents = {t: {} for t in depths}
    total_tokens = 0
    stats = None
    start = time.perf_counter()
    for offset in range(0, len(windows), batch_size):
        b = torch.as_tensor(windows[offset:offset+batch_size].astype(np.int64), device=device)
        _, states = model(b[:,:-1], max(depths), collect=True)
        targets = b[:,1:]
        for t in depths:
            losses = F.cross_entropy(model.readout(states[t]).float().reshape(-1, model.cfg.vocab_size), targets.reshape(-1), reduction='none').reshape_as(targets)
            totals[t] += losses.double().sum().item()
            for j, row_losses in enumerate(losses):
                doc = str(ids[offset+j])
                record = documents[t].setdefault(doc, {'nll_sum':0.0,'tokens':0})
                record['nll_sum'] += row_losses.double().sum().item()
                record['tokens'] += row_losses.numel()
        total_tokens += targets.numel()
        if stats is None:
            updates = [states[t+1]-states[t] for t in range(len(states)-1)]
            stats = []
            e = states[0]
            for t, delta in enumerate(updates):
                h = states[t]
                # Ratio of injected input RMS to the state part of branch input.
                denominator = h.float().square().mean(-1).sqrt()
                if model.cfg.relative or model.cfg.normalize_skip:
                    denominator = torch.ones_like(denominator)
                ratio = (model.alpha.abs()*e.float().square().mean(-1).sqrt() / denominator.clamp_min(1e-8)).mean().item()
                stats.append(dict(loop=t+1, state_rms=h.float().square().mean().sqrt().item(), update_rms=delta.float().square().mean().sqrt().item(),
                    input_state_ratio=ratio, update_cosine_previous=None if t==0 else F.cosine_similarity(delta.float(),updates[t-1].float(),dim=-1).mean().item()))
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return dict(split='validation', tokens=total_tokens, windows=len(windows), documents=len(set(ids)),
        metrics={str(t):dict(nll=totals[t]/total_tokens, byte_ppl=math.exp(totals[t]/total_tokens), bits_per_evaluated_byte=totals[t]/total_tokens/math.log(2)) for t in depths},
        per_document=documents, diagnostics_first_batch=stats, evaluation_seconds=time.perf_counter()-start)


def register(cfg, cfg_path):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    paths = [Path(cfg_path), Path(cfg['protocol']), Path(cfg['data'])/'manifest.json'] + sorted(Path('looped_models').glob('*.py'))
    hashes = {str(p):sha256(p.read_bytes()) for p in paths}
    registration = root/'registration.json'
    if registration.exists():
        old = json.loads(registration.read_text())
        if old['hashes'] != hashes:
            raise ValueError('Registered code/config/protocol/data changed; use a new experiment output ID')
        return old
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), hashes=hashes, config=cfg,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(), device=cfg['device'], threads=cfg['threads']))
    snapshot = root/'source_snapshot'
    for p in paths:
        dest = snapshot/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p,dest)
    write_json(registration,record)
    return record


def train_one(cfg, arm, seed, train, val, ids, registration):
    directory = Path(cfg['output'])/f'{arm}_s{seed}'
    directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():
        return json.loads((directory/'result.json').read_text())
    torch.manual_seed(seed)
    model_cfg = ModelConfig(**{k:cfg[k] for k in ['width','intermediate','heads','kv_heads','core_layers']}, **ARMS[arm])
    model = LoopedLM(model_cfg).to(cfg['device'])
    if model.parameter_count() > 10_000_000:
        raise ValueError('Parameter budget exceeded')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9,0.95), weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(10000+seed)
    loops = 1 if arm=='shallow' else cfg['train_loops']
    checkpoint_path = directory/'last.pt'
    start_step, prior_seconds = 0, 0.0
    if checkpoint_path.exists():
        ck = torch.load(checkpoint_path, map_location=cfg['device'], weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        generator.set_state(ck['data_rng'].cpu()); torch.set_rng_state(ck['torch_rng'].cpu())
        if 'cuda_rng' in ck and cfg['device'].startswith('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
        start_step, prior_seconds = ck['step'], ck['training_seconds']
    start = time.perf_counter()
    elapsed = prior_seconds
    if cfg['device'].startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with (directory/'learning_curve.jsonl').open('a') as log:
        for step in range(start_step, cfg['steps']):
            model.train()
            warmup = cfg['warmup_steps']
            progress = max(0,step-warmup)/max(1,cfg['steps']-1-warmup)
            factor = (step+1)/warmup if step<warmup else 0.1+0.9*0.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups: group['lr']=cfg['lr']*factor
            idx = torch.randint(len(train),(cfg['batch_size'],),generator=generator)
            batch = torch.as_tensor(train[idx.numpy()].astype(np.int64),device=cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch[:,:-1], loops, grad_checkpoint=cfg['grad_checkpoint'])
            loss = F.cross_entropy(logits.reshape(-1,256),batch[:,1:].reshape(-1))
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%cfg['eval_every']==0 or step+1==cfg['steps']:
                if cfg['device'].startswith('cuda'): torch.cuda.synchronize()
                elapsed = prior_seconds+time.perf_counter()-start
                ck = dict(model=model.state_dict(), model_config=model.config_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    data_rng=generator.get_state(), step=step+1, tokens_seen=(step+1)*cfg['batch_size']*cfg['context'],
                    training_seconds=elapsed, config=cfg, arm=arm, seed=seed, provenance=registration['hashes'])
                if cfg['device'].startswith('cuda'): ck['cuda_rng']=torch.cuda.get_rng_state_all()
                atomically_save(checkpoint_path,ck)
                eval_start=time.perf_counter()
                ev=evaluate(model,val,ids,[loops],cfg['device'])
                start += time.perf_counter()-eval_start
                row=dict(step=step+1,tokens=ck['tokens_seen'],train_nll=loss.item(),gradient_norm=float(grad_norm),lr=cfg['lr']*factor,
                    validation_nll=ev['metrics'][str(loops)]['nll'],training_seconds=elapsed)
                log.write(json.dumps(row)+'\n'); log.flush()
    ev=evaluate(model,val,ids,cfg['eval_loops'],cfg['device'])
    result=dict(arm=arm, seed=seed, parameters=model.parameter_count(), train_loops=loops, tokens_seen=cfg['steps']*cfg['batch_size']*cfg['context'],
        training_seconds=elapsed, checkpoint_sha256=sha256(checkpoint_path.read_bytes()), model_config=model.config_dict(), evaluation=ev)
    if cfg['device'].startswith('cuda'): result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    write_json(directory/'result.json',result)
    print(f'{arm} seed={seed}: NLL@{loops}={ev["metrics"][str(loops)]["nll"]:.4f}, {elapsed:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/cpu_pilot.json')
    parser.add_argument('--checkpoint',help='Independent eval of a locally generated checkpoint')
    parser.add_argument('--eval-output',default='reports/independent_eval.json')
    args=parser.parse_args()
    if args.checkpoint:
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        cfg=ck['config']; torch.set_num_threads(cfg['threads'])
        model=LoopedLM(ModelConfig(**ck['model_config'])); model.load_state_dict(ck['model'])
        val,ids=load_windows(cfg['data'],'validation',cfg['eval_windows'])
        result=evaluate(model,val,ids,cfg['eval_loops'])
        result['checkpoint_sha256']=sha256(Path(args.checkpoint).read_bytes())
        Path(args.eval_output).parent.mkdir(parents=True,exist_ok=True)
        write_json(args.eval_output,result); print(args.eval_output)
        return
    cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    if cfg['steps']*cfg['batch_size']*cfg['context'] > 100_000_000: raise ValueError('Token cap exceeded')
    registration=register(cfg,args.config)
    train,_=load_windows(cfg['data'],'train')
    val,ids=load_windows(cfg['data'],'validation',cfg['eval_windows'])
    if train.shape[1]!=cfg['context']+1: raise ValueError('Context mismatch')
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            try: train_one(cfg,arm,seed,train,val,ids,registration)
            except Exception as e:
                write_json(Path(cfg['output'])/f'failure_{arm}_s{seed}.json',dict(error=repr(e),time=datetime.now(timezone.utc).isoformat()))
                raise


if __name__=='__main__': main()
