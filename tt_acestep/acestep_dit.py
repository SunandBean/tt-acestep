# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 Turbo DiT decoder on ONE Blackhole (P100a), TTNN, batch 1.

Mirrors acestep_host.RefDecoder op for op. The block kernels follow the sibling single-P100a Z-Image port
(github.com/SunandBean/tt-z-image-turbo, itself adapted from Tenstorrent's Apache-2.0 Qwen-Image / Z-Image code): fused QKV
minimal_matmul, per-head RMSNorm, adjacent-pair rotary_embedding_llama, fused SwiGLU, fp32-accumulated SDPA.

ACE-Step specifics:
  * GQA (16 query / 8 KV heads) straight into SDPA; sliding-window layers (|i - j| <= 128) and padded keys use
    additive masks
  * AdaLN rows of each timestep are uploaded once: rms(x) * gamma + shift, then gate * branch
  * cross-attention K/V are computed on the host once per song (acestep_host.RefDecoder.encode) and stay on device
  * patch in/out (Conv1d / ConvTranspose1d, k = s = 2) are matmuls on [S_pad, 384] / [S_pad, 128] rows
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from . import acestep_host as host

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover - only on hosts without tt-metal models/
    get_matmul_config = None

MEM = ttnn.DRAM_MEMORY_CONFIG
TURBO_SCHEDULE = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125)  # 8 steps, shift 1, t_r = t (the pipeline's)


def open_device(l1_small_size: int = 98304):  # convs (VAE) keep their config tensors in L1_SMALL
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1),
                                 dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
                                 l1_small_size=l1_small_size)


def close_device(dev):
    ttnn.close_mesh_device(dev)


def dram_stats(dev) -> dict:
    try:
        s = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return {"allocated_bytes": s.total_bytes_allocated_per_bank * s.num_banks,
                "free_bytes": s.total_bytes_free_per_bank * s.num_banks}
    except Exception:  # pragma: no cover - diagnostics only
        return {}


@dataclass
class Precision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    sdpa_fp32_acc: bool = True


def _dev(dev, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor):
    """[D] -> [1, 1, 1, D] bf16 row (norm gamma, shift, gate, bias)."""
    return _dev(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1))


class Layer:
    def __init__(self, dev, w: host.LayerWeights, prec: Precision):
        wd = prec.weight_dtype
        self.wqkv = _dev(dev, w.wqkv.to(torch.bfloat16), wd)
        self.q_norm, self.k_norm = _row(dev, w.q_norm), _row(dev, w.k_norm)
        self.wo = _dev(dev, w.wo.to(torch.bfloat16), wd)
        self.cq = _dev(dev, w.cq.to(torch.bfloat16), wd)
        self.cq_norm = _row(dev, w.cq_norm)
        self.cwo = _dev(dev, w.cwo.to(torch.bfloat16), wd)
        self.cross_norm = _row(dev, w.cross_norm)
        self.w_gateup = _dev(dev, w.w_gateup.to(torch.bfloat16), wd)
        self.w_down = _dev(dev, w.w_down.to(torch.bfloat16), wd)


@dataclass
class Song:
    """Device state of one generation: geometry tables, masks, cross K/V."""

    geo: host.Geometry
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    full_mask: ttnn.Tensor
    sliding_mask: ttnn.Tensor
    cross_mask: ttnn.Tensor
    kv: List[tuple]


class AceStepDiT:
    def __init__(self, dev, ckpt: host.Checkpoint, prec: Optional[Precision] = None):
        self.dev, self.cfg = dev, ckpt.cfg
        self.prec = prec or Precision()
        self.ref = host.RefDecoder(ckpt)  # host weights: layouts, time rows and the cross K/V projection
        self.grid = dev.compute_with_storage_grid_size()
        self.layers = [Layer(dev, w, self.prec) for w in self.ref.weights]
        self.w_in, self.b_in = _dev(dev, self.ref.w_in.to(torch.bfloat16)), _row(dev, self.ref.b_in)
        self.w_out, self.b_out = _dev(dev, self.ref.w_out.to(torch.bfloat16)), _row(dev, self.ref.b_out)
        self.trans_mat = _dev(dev, host.rot_transformation_mat())
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self.ck_rope = ck(ttnn.MathFidelity.HiFi4, True)
        self._mm_cfg_cache: Dict[tuple, object] = {}
        self._rows: Dict[tuple, dict] = {}
        self.sdpa_chunks = (256, 128, 64, 32)
        # The schedule's AdaLN rows go on the device now, before any song's tensors: allocated during the first step
        # they would land behind that song's tables, masks and K/V, so every later allocation (the VAE's conv
        # configs, whose addresses end up in its compiled kernels) would move with the song length and recompile.
        for t in TURBO_SCHEDULE:
            self.rows(t, t)

    # ------------------------------------------------------------------ conditioning
    def rows(self, t: float, t_r: float) -> dict:
        key = (t, t_r)
        if key not in self._rows:
            r = self.ref.rows(t, t_r)
            self._rows[key] = {"layers": [{k: _row(self.dev, v) for k, v in lr.items()} for lr in r.layers],
                               "out_gamma": _row(self.dev, r.out_gamma), "out_shift": _row(self.dev, r.out_shift)}
        return self._rows[key]

    def prepare(self, encoder_hidden_states: torch.Tensor, frames: int) -> Song:
        """[L, 2048] encoder states for a song of `frames` latent frames -> device tables, masks and cross K/V."""
        c = self.cfg
        geo = host.Geometry(frames=frames, enc_len=encoder_hidden_states.shape[0])
        cos, sin = host.rope_tables(geo.seq_pad, c)
        full, sliding = host.self_masks(geo, c.sliding_window)
        kv = [(_dev(self.dev, k[None].to(torch.bfloat16)), _dev(self.dev, v[None].to(torch.bfloat16)))
              for k, v in self.ref.encode(encoder_hidden_states.float(), geo)]
        f = lambda m: _dev(self.dev, m.to(torch.bfloat16)[None, None])
        return Song(geo, f(cos), f(sin), f(full), f(sliding), f(host.cross_mask(geo)), kv)

    def release(self, song: Song):
        for t in (song.cos, song.sin, song.full_mask, song.sliding_mask, song.cross_mask):
            ttnn.deallocate(t)
        for k, v in song.kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)

    # ------------------------------------------------------------------ ops
    def _mm(self, x, w, M, K, N, fuse_swiglu=False):
        key = (M, K, N)
        if key not in self._mm_cfg_cache:
            self._mm_cfg_cache[key] = get_matmul_config(M, K, N, self.grid)
        return ttnn.experimental.minimal_matmul(x, w, config=self._mm_cfg_cache[key], compute_kernel_config=self.ck_mm,
                                                dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=fuse_swiglu)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.cfg.eps, weight=w, compute_kernel_config=self.ck_norm, memory_config=MEM)

    def _sdpa(self, q, k, v, mask, Sq, Sk):
        chunk = lambda n: next(c for c in self.sdpa_chunks if n % c == 0)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=chunk(Sq),
                                     k_chunk_size=chunk(Sk), exp_approx_mode=False)
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=self.cfg.head_dim ** -0.5, program_config=cfg,
            compute_kernel_config=self.ck_sdpa, memory_config=MEM)

    def _heads(self, x, S, n):
        """[1, 1, S, n * D] -> [1, n, S, D]."""
        x = ttnn.reshape(x, [1, S, n, self.cfg.head_dim])
        return ttnn.permute(x, (0, 2, 1, 3), memory_config=MEM)

    def _modulated(self, x, gamma, shift):
        n = self._rms(x, gamma)
        out = ttnn.add(n, shift, memory_config=MEM)
        ttnn.deallocate(n)
        return out

    def _gated_add(self, x, branch, gate):
        g = ttnn.multiply(branch, gate, memory_config=MEM) if gate is not None else branch
        out = ttnn.add(x, g, memory_config=MEM)
        if gate is not None:
            ttnn.deallocate(g)
        ttnn.deallocate(branch)
        ttnn.deallocate(x)
        return out

    def _self_attention(self, h, L: Layer, S, song: Song, sliding: bool):
        c = self.cfg
        qkv = self._mm(h, L.wqkv, S, c.hidden, (c.heads + 2 * c.kv_heads) * c.head_dim)
        if len(qkv.shape) != 4:
            qkv = ttnn.reshape(qkv, [1, 1, S, qkv.shape[-1]])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.kv_heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        roped = []
        for t, w in ((q, L.q_norm), (k, L.k_norm)):
            n = self._rms(t, w)
            ttnn.deallocate(t)
            roped.append(ttnn.experimental.rotary_embedding_llama(n, song.cos, song.sin, self.trans_mat,
                                                                  is_decode_mode=False, compute_kernel_config=self.ck_rope))
            ttnn.deallocate(n)
        a = self._sdpa(roped[0], roped[1], v, song.sliding_mask if sliding else song.full_mask, S, S)
        for t in (*roped, v):
            ttnn.deallocate(t)
        a2 = ttnn.transformer.concatenate_heads(a, memory_config=MEM)
        ttnn.deallocate(a)
        o = self._mm(a2, L.wo, S, c.hidden, c.hidden)
        ttnn.deallocate(a2)
        return o

    def _cross_attention(self, h, L: Layer, S, song: Song, i: int):
        c = self.cfg
        q = self._mm(h, L.cq, S, c.hidden, c.hidden)
        if len(q.shape) != 4:
            q = ttnn.reshape(q, [1, 1, S, c.hidden])
        qh = self._heads(q, S, c.heads)
        ttnn.deallocate(q)
        qn = self._rms(qh, L.cq_norm)
        ttnn.deallocate(qh)
        k, v = song.kv[i]
        a = self._sdpa(qn, k, v, song.cross_mask, S, song.geo.enc_pad)
        ttnn.deallocate(qn)
        a2 = ttnn.transformer.concatenate_heads(a, memory_config=MEM)
        ttnn.deallocate(a)
        o = self._mm(a2, L.cwo, S, c.hidden, c.hidden)
        ttnn.deallocate(a2)
        return o

    # ------------------------------------------------------------------ public
    def forward(self, x_t: torch.Tensor, t: float, t_r: float, context: torch.Tensor, song: Song,
                taps: Optional[list] = None) -> torch.Tensor:
        """[T, 64] noisy latents + [T, 128] context -> [T, 64] float32 velocity."""
        c, geo = self.cfg, song.geo
        S = geo.seq_pad
        rows = self.rows(t, t_r)
        p = _dev(self.dev, host.patchify(context.float(), x_t.float(), geo).to(torch.bfloat16).reshape(1, 1, S, -1))
        x = ttnn.linear(p, self.w_in, bias=self.b_in, compute_kernel_config=self.ck_mm, memory_config=MEM,
                        dtype=ttnn.bfloat16)
        ttnn.deallocate(p)
        tap = (lambda v: taps.append(ttnn.to_torch(v)[0, 0].float())) if taps is not None else (lambda v: None)
        for i, L in enumerate(self.layers):
            r = rows["layers"][i]
            h = self._modulated(x, r["self_gamma"], r["self_shift"])
            o = self._self_attention(h, L, S, song, c.sliding(i))
            ttnn.deallocate(h)
            x = self._gated_add(x, o, r["gate"])
            h = self._rms(x, L.cross_norm)
            o = self._cross_attention(h, L, S, song, i)
            ttnn.deallocate(h)
            x = self._gated_add(x, o, None)
            h = self._modulated(x, r["mlp_gamma"], r["mlp_shift"])
            m = self._mm(h, L.w_gateup, S, c.hidden, 2 * c.intermediate, fuse_swiglu=True)
            ttnn.deallocate(h)
            d = self._mm(m, L.w_down, S, c.intermediate, c.hidden)
            ttnn.deallocate(m)
            x = self._gated_add(x, d, r["c_gate"])
            tap(x)
        h = self._modulated(x, rows["out_gamma"], rows["out_shift"])
        ttnn.deallocate(x)
        out = ttnn.linear(h, self.w_out, bias=self.b_out, compute_kernel_config=self.ck_mm, memory_config=MEM,
                          dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        host_out = ttnn.to_torch(out)[0, 0, :S].float()
        ttnn.deallocate(out)
        return host.unpatchify(host_out, geo, c.out_channels)
