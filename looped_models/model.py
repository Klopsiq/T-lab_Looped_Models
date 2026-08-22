"""Compact Qwen-style causal block with recurrent weight sharing."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 256
    width: int = 64
    intermediate: int = 128
    heads: int = 4
    kv_heads: int = 2
    core_layers: int = 2
    relative: bool = False
    depth_rope: bool = False
    normalize_skip: bool = False
    mlp_first: bool = False
    depth_theta: float = 1000.0
    position_theta: float = 10000.0


def unit_rms(x):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-8).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def rotate(x, position, theta, dimensions=None, inverse=False):
    """Split-half RoPE, generated for the requested position; no depth clamp."""
    dimensions = dimensions or x.shape[-1]
    if dimensions % 2:
        raise ValueError("Rotated dimensions must be even")
    half = dimensions // 2
    freq = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.as_tensor(position, device=x.device, dtype=torch.float32)[..., None] * freq
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    if inverse:
        s = -s
    a, b = x[..., :dimensions].chunk(2, dim=-1)
    return torch.cat((a*c - b*s, a*s + b*c, x[..., dimensions:]), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, self.head_dim = cfg.width, cfg.width // cfg.heads
        if d % cfg.heads or cfg.heads % cfg.kv_heads or self.head_dim % 2:
            raise ValueError("Invalid head geometry")
        self.cfg = cfg
        self.attn_norm, self.mlp_norm = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, cfg.heads*self.head_dim, bias=False)
        self.k = nn.Linear(d, cfg.kv_heads*self.head_dim, bias=False)
        self.v = nn.Linear(d, cfg.kv_heads*self.head_dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out = nn.Linear(d, d, bias=False)
        self.gate, self.up = nn.Linear(d, cfg.intermediate, bias=False), nn.Linear(d, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, d, bias=False)

    def attention(self, h):
        batch, length, width = h.shape
        x = self.attn_norm(h)
        def split(t, heads):
            return t.reshape(batch, length, heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(split(self.q(x), self.cfg.heads))
        k = self.k_norm(split(self.k(x), self.cfg.kv_heads))
        v = split(self.v(x), self.cfg.kv_heads)
        positions = torch.arange(length, device=h.device)
        q, k = rotate(q, positions, self.cfg.position_theta), rotate(k, positions, self.cfg.position_theta)
        repeats = self.cfg.heads // self.cfg.kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return h + self.out(a.transpose(1, 2).reshape(batch, length, width))

    def mlp(self, h):
        x = self.mlp_norm(h)
        return h + self.down(F.silu(self.gate(x))*self.up(x))

    def forward(self, h):
        if self.cfg.mlp_first:
            return self.attention(self.mlp(h))
        return self.mlp(self.attention(h))


class LoopedLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.embed_norm, self.final_norm = RMSNorm(cfg.width), RMSNorm(cfg.width)
        self.core = nn.Sequential(*(Block(cfg) for _ in range(cfg.core_layers)))
        self.alpha = nn.Parameter(torch.ones(()))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        if cfg.width % 4:
            raise ValueError("Half-channel depth rotation requires width divisible by four")

    def step(self, h, e, t):
        skip = unit_rms(h) if self.cfg.normalize_skip else h
        x = unit_rms(skip) if self.cfg.relative else skip
        x = x + self.alpha*e
        if self.cfg.depth_rope:
            x = rotate(x, t, self.cfg.depth_theta, self.cfg.width//2)
        delta = self.core(x) - x
        if self.cfg.depth_rope:
            delta = rotate(delta, t, self.cfg.depth_theta, self.cfg.width//2, inverse=True)
        return skip + delta

    def readout(self, h):
        # Tied by construction: no separately allocated output matrix.
        return F.linear(self.final_norm(h), self.embedding.weight)

    def forward(self, tokens, loops=4, collect=False, grad_checkpoint=False):
        if loops < 1:
            raise ValueError("loops must be positive")
        e = self.embed_norm(self.embedding(tokens))
        h = e
        trajectory = [h] if collect else None
        for t in range(loops):
            if grad_checkpoint and self.training:
                h = checkpoint(self.step, h, e, t, use_reentrant=False)
            else:
                h = self.step(h, e, t)
            if collect:
                trajectory.append(h)
        return (self.readout(h), trajectory) if collect else self.readout(h)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.cfg)
