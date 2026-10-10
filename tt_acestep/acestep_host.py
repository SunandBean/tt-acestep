"""Host (torch, CPU) side of the ACE-Step 1.5 Turbo DiT port to one Tenstorrent P100a.

Everything the TT decoder needs that is cheap or exact on the host: weight layouts (fused QKV, rotate-half ->
adjacent-pair permutation for rotary_embedding_llama, fused SwiGLU, patch in/out as matmuls), the per-timestep
AdaLN rows (the turbo schedule is fixed, so they are computed once and folded into the RMSNorm weights), RoPE
tables and attention masks for the padded sequence.

RefDecoder runs the same formulation in float32 on the CPU. It must match the upstream AceStepDiTModel before the
TT decoder is compared with it, so a layout mistake shows up without a device.

Upstream (acestep/models/turbo/modeling_acestep_v15_turbo.py, AceStepDiTModel.forward) as used by the pipeline:
  temb, proj = time_embed(t) + time_embed_r(t - r)      (r = t: the second term is a constant)
  x = proj_in(cat(context_latents, x_t))                Conv1d k=2 s=2: 192 -> 2048, S = ceil(T / 2)
  enc = condition_embedder(encoder_hidden_states)
  24 x layer: x += gate * self_attn(rms(x)*(1+scale)+shift)   (RoPE, sliding window |i-j| <= 128 on even layers)
              x += cross_attn(rms(x), enc)                     (no mask, no RoPE)
              x += c_gate * mlp(rms(x)*(1+c_scale)+c_shift)
  out = proj_out(rms(x)*(1+scale)+shift)[:T]            ConvTranspose1d k=2 s=2: 2048 -> 64
No padding mask is applied upstream (attention_mask and encoder_attention_mask are reset to None).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

TILE = 32
SEQ_MULTIPLE = 128  # sequence lengths are padded to multiples of this at least (SDPA chunk sizes divide it)
# Every distinct padded length compiles its own kernels on the card once. Coarse buckets keep that to a few shapes
# that can be compiled ahead of time (warm_cache.py); the padding is masked out, so results do not change.
SEQ_BUCKET = 512  # DiT patches: 30-480 s songs (375-6000 patches) -> 12 shapes
ENC_BUCKET = 512  # encoder tokens (prompt + lyrics): a few shapes whatever the lyrics length
NEG = -1e9  # additive mask value for excluded keys (finite, representable in bf16)


@dataclass(frozen=True)
class Config:
    hidden: int = 2048
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 6144
    layers: int = 24
    layer_types: tuple = ()
    sliding_window: int = 128
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    in_channels: int = 192
    out_channels: int = 64
    patch: int = 2

    @classmethod
    def load(cls, checkpoint_dir) -> "Config":
        c = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        return cls(hidden=c["hidden_size"], heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                   head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                   intermediate=c["intermediate_size"], layers=c["num_hidden_layers"],
                   layer_types=tuple(c["layer_types"]), sliding_window=c["sliding_window"] or 128,
                   rope_theta=c["rope_theta"], eps=c["rms_norm_eps"], in_channels=c["in_channels"],
                   out_channels=c["audio_acoustic_hidden_dim"], patch=c["patch_size"])

    def sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"


def round_up(n: int, m: int = SEQ_MULTIPLE) -> int:
    return (n + m - 1) // m * m


class Checkpoint:
    """decoder.* tensors of model.safetensors, read on demand."""

    def __init__(self, checkpoint_dir):
        from safetensors import safe_open

        self.dir = Path(checkpoint_dir)
        self.cfg = Config.load(self.dir)
        self._file = safe_open(str(self.dir / "model.safetensors"), framework="pt")

    def get(self, name: str, dtype=torch.float32) -> torch.Tensor:
        return self._file.get_tensor(f"decoder.{name}").to(dtype)


# ----------------------------------------------------------------------------- weight layouts
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: rotate-half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1).

    Applying it to the rows of q_proj/k_proj (per head) and to q_norm/k_norm leaves every q.k product unchanged and
    lets tt's rotary_embedding_llama (adjacent pairs) replace upstream's rotate_half RoPE."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    return w_out_in.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """[K, 2N] weight for minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p+1 = up tile p."""
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    K, N = g.shape
    assert N % tile == 0, N
    return torch.stack([g.view(K, N // tile, tile), u.view(K, N // tile, tile)], dim=2).reshape(K, 2 * N).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    """[1, 1, 32, 32] T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent-pair rotation)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


@dataclass
class LayerWeights:
    """One DiT layer in the TT layout (float32 host tensors; the device copy is bf16)."""

    wqkv: torch.Tensor  # [2048, 2048 + 1024 + 1024] self-attn q|k|v, q/k rows permuted to adjacent pairs
    q_norm: torch.Tensor  # [128] permuted
    k_norm: torch.Tensor  # [128] permuted
    wo: torch.Tensor  # [2048, 2048]
    cq: torch.Tensor  # [2048, 2048] cross-attn q (no RoPE: natural order)
    ckv: torch.Tensor  # [2048, 1024 + 1024] cross-attn k|v applied to the encoder states
    cq_norm: torch.Tensor
    ck_norm: torch.Tensor
    cwo: torch.Tensor
    cross_norm: torch.Tensor  # [2048] cross_attn_norm (not modulated)
    w_gateup: torch.Tensor  # [2048, 2 * 6144] tile-interleaved gate|up
    w_down: torch.Tensor  # [6144, 2048]
    self_norm: torch.Tensor  # [2048] self_attn_norm, folded with (1 + scale) per step
    mlp_norm: torch.Tensor  # [2048] mlp_norm, folded with (1 + c_scale) per step
    table: torch.Tensor  # [6, 2048] scale_shift_table


def layer_weights(ckpt: Checkpoint, i: int) -> LayerWeights:
    c = ckpt.cfg
    g = lambda k: ckpt.get(f"layers.{i}.{k}")
    perm = interleave_pairs_permutation(c.head_dim)
    wq = permute_heads_rows(g("self_attn.q_proj.weight"), c.heads, c.head_dim, perm)
    wk = permute_heads_rows(g("self_attn.k_proj.weight"), c.kv_heads, c.head_dim, perm)
    return LayerWeights(
        wqkv=torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(g("self_attn.v_proj.weight"))], dim=1),
        q_norm=g("self_attn.q_norm.weight")[perm], k_norm=g("self_attn.k_norm.weight")[perm],
        wo=linear_to_mm(g("self_attn.o_proj.weight")),
        cq=linear_to_mm(g("cross_attn.q_proj.weight")),
        ckv=torch.cat([linear_to_mm(g("cross_attn.k_proj.weight")), linear_to_mm(g("cross_attn.v_proj.weight"))], dim=1),
        cq_norm=g("cross_attn.q_norm.weight"), ck_norm=g("cross_attn.k_norm.weight"),
        cwo=linear_to_mm(g("cross_attn.o_proj.weight")), cross_norm=g("cross_attn_norm.weight"),
        w_gateup=swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")),
        w_down=linear_to_mm(g("mlp.down_proj.weight")),
        self_norm=g("self_attn_norm.weight"), mlp_norm=g("mlp_norm.weight"), table=g("scale_shift_table")[0])


def conv_patch_matrix(w: torch.Tensor) -> torch.Tensor:
    """Conv1d weight [out, in, k] (stride k) -> matmul weight [k * in, out] for rows ordered (k, channel)."""
    return w.permute(0, 2, 1).reshape(w.shape[0], -1).t().contiguous()


def conv_transpose_patch_matrix(w: torch.Tensor, b: torch.Tensor):
    """ConvTranspose1d weight [in, out, k] (stride k) -> matmul weight [in, k * out] and bias [k * out],
    columns ordered (k, channel)."""
    return w.permute(0, 2, 1).reshape(w.shape[0], -1).contiguous(), b.repeat(w.shape[2])


def patch_in_weight(ckpt: Checkpoint):
    return conv_patch_matrix(ckpt.get("proj_in.1.weight")), ckpt.get("proj_in.1.bias")


def patch_out_weight(ckpt: Checkpoint):
    return conv_transpose_patch_matrix(ckpt.get("proj_out.1.weight"), ckpt.get("proj_out.1.bias"))


# ----------------------------------------------------------------------------- sequence layout
@dataclass(frozen=True)
class Geometry:
    """T latent frames -> S = ceil(T/2) patches padded to S_pad; L encoder tokens padded to L_pad."""

    frames: int
    enc_len: int

    @property
    def seq(self) -> int:
        return (self.frames + 1) // 2

    @property
    def seq_pad(self) -> int:
        return round_up(self.seq, SEQ_BUCKET)

    @property
    def enc_pad(self) -> int:
        return round_up(self.enc_len, ENC_BUCKET)


def patchify(context_latents: torch.Tensor, x_t: torch.Tensor, geo: Geometry) -> torch.Tensor:
    """[T, 128] context + [T, 64] noisy latents -> [S_pad, 384] patch rows (zeros past T, like upstream F.pad)."""
    x = torch.cat([context_latents, x_t], dim=-1)
    rows = torch.zeros(geo.seq_pad * 2, x.shape[-1], dtype=x.dtype)
    rows[: geo.frames] = x
    return rows.reshape(geo.seq_pad, -1)


def unpatchify(out: torch.Tensor, geo: Geometry, channels: int = 64) -> torch.Tensor:
    """[S_pad, 128] (k, channel) columns -> [T, 64]."""
    return out.reshape(geo.seq_pad * 2, channels)[: geo.frames]


def rope_tables(seq_pad: int, cfg: Config, dtype=torch.float32):
    """[S_pad, 128] cos/sin for positions 0..S_pad-1 in the adjacent-pair layout (Qwen3 RoPE, theta 1e6)."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim))
    ang = torch.outer(torch.arange(seq_pad, dtype=torch.float64), inv).float()
    return torch.cos(ang).repeat_interleave(2, -1).to(dtype), torch.sin(ang).repeat_interleave(2, -1).to(dtype)


def self_masks(geo: Geometry, window: int):
    """Additive [S_pad, S_pad] masks (full, sliding): keys past S excluded; sliding keeps |i - j| <= window."""
    i = torch.arange(geo.seq_pad)[:, None]
    j = torch.arange(geo.seq_pad)[None, :]
    valid = j < geo.seq
    full = torch.where(valid, 0.0, NEG).expand(geo.seq_pad, -1).contiguous()
    sliding = torch.where(valid & ((i - j).abs() <= window), 0.0, NEG)
    return full, sliding


def cross_mask(geo: Geometry):
    """Additive [S_pad, L_pad] mask excluding padded encoder tokens."""
    valid = torch.arange(geo.enc_pad)[None, :] < geo.enc_len
    return torch.where(valid, 0.0, NEG).expand(geo.seq_pad, -1).contiguous()


# ----------------------------------------------------------------------------- time conditioning
def _timestep_embedding(t: float, dim: int = 256, max_period: float = 10000.0, scale: float = 1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.tensor([t * scale], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _time_mlp(ckpt: Checkpoint, prefix: str, t: float):
    g = lambda k: ckpt.get(f"{prefix}.{k}")
    f = torch.nn.functional
    h = f.linear(_timestep_embedding(t), g("linear_1.weight"), g("linear_1.bias"))
    temb = f.linear(f.silu(h), g("linear_2.weight"), g("linear_2.bias"))
    proj = f.linear(f.silu(temb), g("time_proj.weight"), g("time_proj.bias"))
    return temb[0], proj[0].reshape(6, -1)


@dataclass
class StepRows:
    """Per-timestep AdaLN rows, folded: rms(x) * w * (1 + scale) + shift == rms(x) * gamma + shift."""

    layers: List[Dict[str, torch.Tensor]]  # per layer: self_gamma, self_shift, gate, mlp_gamma, mlp_shift, c_gate
    out_gamma: torch.Tensor
    out_shift: torch.Tensor


def step_rows(ckpt: Checkpoint, weights: List[LayerWeights], t: float, t_r: float) -> StepRows:
    temb_t, proj_t = _time_mlp(ckpt, "time_embed", t)
    temb_r, proj_r = _time_mlp(ckpt, "time_embed_r", t - t_r)
    temb, proj = temb_t + temb_r, proj_t + proj_r
    layers = []
    for w in weights:
        shift, scale, gate, c_shift, c_scale, c_gate = (w.table + proj).unbind(0)
        layers.append({"self_gamma": w.self_norm * (1 + scale), "self_shift": shift, "gate": gate,
                       "mlp_gamma": w.mlp_norm * (1 + c_scale), "mlp_shift": c_shift, "c_gate": c_gate})
    shift, scale = (ckpt.get("scale_shift_table")[0] + temb[None]).unbind(0)
    return StepRows(layers, ckpt.get("norm_out.weight") * (1 + scale), shift)


# ----------------------------------------------------------------------------- float reference
def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


def attention(q, k, v, mask):
    """q [H, S, D], k/v [Hkv, Sk, D] (GQA), additive mask [S, Sk]."""
    rep = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5 + mask
    return torch.softmax(scores, -1) @ v


class RefDecoder:
    """float32 CPU decoder in the TT formulation (padding, masks, fused and permuted weights, folded AdaLN)."""

    def __init__(self, ckpt: Checkpoint):
        self.ckpt, self.cfg = ckpt, ckpt.cfg
        self.weights = [layer_weights(ckpt, i) for i in range(self.cfg.layers)]
        self.w_in, self.b_in = patch_in_weight(ckpt)
        self.w_out, self.b_out = patch_out_weight(ckpt)
        self.w_cond = linear_to_mm(ckpt.get("condition_embedder.weight"))
        self.b_cond = ckpt.get("condition_embedder.bias")
        self._rows: Dict[tuple, StepRows] = {}

    def rows(self, t: float, t_r: float) -> StepRows:
        if (t, t_r) not in self._rows:
            self._rows[(t, t_r)] = step_rows(self.ckpt, self.weights, t, t_r)
        return self._rows[(t, t_r)]

    def encode(self, enc: torch.Tensor, geo: Geometry):
        """[L, 2048] encoder states -> per-layer cross K/V [Hkv, L_pad, D] (padded rows are masked)."""
        c = self.cfg
        e = torch.zeros(geo.enc_pad, enc.shape[-1])
        e[: geo.enc_len] = enc
        e = e @ self.w_cond + self.b_cond
        kv = []
        for w in self.weights:
            k, v = (e @ w.ckv).split(c.kv_heads * c.head_dim, dim=-1)
            k = rms(k.view(-1, c.kv_heads, c.head_dim), w.ck_norm, c.eps).transpose(0, 1)
            kv.append((k, v.view(-1, c.kv_heads, c.head_dim).transpose(0, 1)))
        return kv

    def forward(self, x_t: torch.Tensor, t: float, t_r: float, context: torch.Tensor, kv, geo: Geometry,
                taps: list = None):
        """[T, 64] noisy latents -> [T, 64] velocity; taps (optional) receives the hidden state after each layer."""
        c, rows = self.cfg, self.rows(t, t_r)
        cos, sin = rope_tables(geo.seq_pad, c)
        full, sliding = self_masks(geo, c.sliding_window)
        xmask = cross_mask(geo)
        x = patchify(context, x_t, geo) @ self.w_in + self.b_in
        for i, w in enumerate(self.weights):
            r = rows.layers[i]
            h = rms(x, r["self_gamma"], c.eps) + r["self_shift"]
            q, k, v = (h @ w.wqkv).split([c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim], -1)
            q = apply_rope_adjacent(rms(q.view(-1, c.heads, c.head_dim), w.q_norm, c.eps).transpose(0, 1), cos, sin)
            k = apply_rope_adjacent(rms(k.view(-1, c.kv_heads, c.head_dim), w.k_norm, c.eps).transpose(0, 1), cos, sin)
            v = v.view(-1, c.kv_heads, c.head_dim).transpose(0, 1)
            a = attention(q, k, v, sliding if c.sliding(i) else full).transpose(0, 1).reshape(geo.seq_pad, -1)
            x = x + (a @ w.wo) * r["gate"]
            h = rms(x, w.cross_norm, c.eps)
            q = rms((h @ w.cq).view(-1, c.heads, c.head_dim), w.cq_norm, c.eps).transpose(0, 1)
            a = attention(q, *kv[i], xmask).transpose(0, 1).reshape(geo.seq_pad, -1)
            x = x + a @ w.cwo
            h = rms(x, r["mlp_gamma"], c.eps) + r["mlp_shift"]
            gu = (h @ w.w_gateup).view(geo.seq_pad, -1, 2, TILE)
            x = x + ((torch.nn.functional.silu(gu[:, :, 0]) * gu[:, :, 1]).reshape(geo.seq_pad, -1) @ w.w_down) * r["c_gate"]
            if taps is not None:
                taps.append(x.clone())
        x = rms(x, rows.out_gamma, c.eps) + rows.out_shift
        return unpatchify(x @ self.w_out + self.b_out, geo, c.out_channels)
