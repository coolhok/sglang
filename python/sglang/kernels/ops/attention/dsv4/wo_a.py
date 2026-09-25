from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit
from sglang.kernels.ops.layernorm.mxfp8_epilogue import ue8m0_scale

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GROUPS = 2
RANK = 1024
N_OUT = GROUPS * RANK
SCALE_BYTES = 8192
MAX_M = 32
SM90_MAX_M = 2


@triton.jit
def _fused_rope_wo_a_bf16_sm90_kernel(
    X,
    W,
    FREQS,
    POSITIONS,
    Y,
    stride_xt,
    stride_xg,
    stride_wg,
    stride_wr,
    stride_wd,
    stride_yt,
    stride_yg,
    stride_yr,
    R: tl.constexpr,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    row_block = tl.program_id(1)
    group = tl.program_id(2)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = tl.arange(0, D)

    x = tl.load(X + token * stride_xt + group * stride_xg + columns).to(tl.float32)
    head_column = columns % HEAD_DIM
    rope_column = head_column - (HEAD_DIM - ROPE_DIM)
    is_rope = rope_column >= 0
    partner = tl.load(
        X + token * stride_xt + group * stride_xg + (columns ^ 1),
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)
    pair = tl.where(is_rope, rope_column // 2, 0)
    position = tl.load(POSITIONS + token)
    freq_offsets = position * ROPE_DIM + pair * 2
    cosine = tl.load(FREQS + freq_offsets, mask=is_rope, other=1.0)
    sine = tl.load(FREQS + freq_offsets + 1, mask=is_rope, other=0.0)
    rotated = tl.where(
        (rope_column & 1) == 0,
        x * cosine + partner * sine,
        x * cosine - partner * sine,
    )
    # Match the unfused path's BF16 materialization between inverse RoPE and WO-A.
    x = tl.where(is_rope, rotated.to(tl.bfloat16), x).to(tl.float32)

    weight = tl.load(
        W
        + group * stride_wg
        + rows[:, None] * stride_wr
        + columns[None, :] * stride_wd,
        mask=rows[:, None] < R,
        other=0.0,
    ).to(tl.float32)
    result = tl.sum(weight * x[None, :], axis=1)
    tl.store(
        Y + token * stride_yt + group * stride_yg + rows * stride_yr,
        result,
        mask=rows < R,
    )


def fused_rope_wo_a_bf16_sm90(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Inverse RoPE + grouped BF16 WO-A for the DSV4.1 TP8 decode shape."""
    assert x.ndim == weight.ndim == 3
    assert x.dtype == weight.dtype == torch.bfloat16
    assert freqs_cis.dtype == torch.float32 and freqs_cis.ndim == 2
    assert positions.dtype in (torch.int32, torch.int64) and positions.ndim == 1
    assert x.is_cuda and x.device == weight.device == freqs_cis.device
    assert x.device == positions.device
    tokens, groups, dim = x.shape
    weight_groups, rows, weight_dim = weight.shape
    assert 0 < tokens <= SM90_MAX_M
    assert positions.shape[0] == tokens
    assert (groups, rows, dim) == (1, 1024, 4096)
    assert (weight_groups, weight_dim) == (groups, dim)
    assert freqs_cis.shape[1] == 64
    assert freqs_cis.is_contiguous() and positions.is_contiguous()
    assert x.stride(2) == weight.stride(2) == 1

    output = torch.empty((tokens, groups, rows), dtype=x.dtype, device=x.device)
    block_n = 4
    _fused_rope_wo_a_bf16_sm90_kernel[(tokens, triton.cdiv(rows, block_n), groups)](
        x,
        weight,
        freqs_cis,
        positions,
        output,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        R=rows,
        D=dim,
        HEAD_DIM=512,
        ROPE_DIM=64,
        BLOCK_N=block_n,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@cache_once
def _jit_module(max_tokens: int) -> Module:
    return load_jit(
        f"fused_rope_wo_a_m{max_tokens}",
        cuda_files=["deepseek_v4/wo_a_fused.cuh"],
        cuda_wrappers=[("run", f"wo_a_fused_run<{max_tokens}>")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def fused_rope_wo_a_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    out_mxfp8: bool = True,
) -> Sequence[torch.Tensor]:
    """Inverse RoPE + BF16 WO-A, returning BF16 or FlashInfer-swizzled MXFP8."""
    num_tokens = x.shape[0]
    assert num_tokens <= MAX_M
    max_tokens = 16 if num_tokens <= 16 else 32
    module = _jit_module(max_tokens)
    if out_mxfp8:
        q = torch.empty((num_tokens, N_OUT), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty(SCALE_BYTES, dtype=torch.uint8, device=x.device)
        module.run(x, weight, freqs_cis, positions, q, scales, None)
        return q, scales
    else:
        y = x.new_empty((num_tokens, N_OUT))
        module.run(x, weight, freqs_cis, positions, None, None, y)
        return (y,)


@triton.jit
def _wo_a_bf16_gemv_kernel(X, W, Y, R: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, D)
    x = tl.load(X + group * D + columns).to(tl.float32)
    w = tl.load(
        W + (group * R + rows[:, None]) * D + columns[None, :],
        rows[:, None] < R,
        0,
    ).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + group * R + rows, result, rows < R)


def wo_a_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for one token."""
    assert x.shape[0] == 1 and x.ndim == weight.ndim == 3
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.is_contiguous() and weight.is_contiguous()
    groups, rows, dim = weight.shape
    assert x.shape[1:] == (groups, dim) and dim == triton.next_power_of_2(dim)
    result = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    # One output row per CTA keeps register use low and exposes enough
    # independent weight loads for single-token decode.
    _wo_a_bf16_gemv_kernel[(rows, groups)](
        x,
        weight,
        result,
        rows,
        dim,
        1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@triton.jit
def _wo_a_partial(X, W, P, M: tl.constexpr, SX: tl.constexpr):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


@triton.jit
def _wo_a_reduce(P, Y, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_bf16_small_batch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for the TP4 WO-A shape.
    Partial sums stay in FP32 until the final BF16, token-major store."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce[(triton.cdiv(m * 2048, 256),)](partial, result, m * 2048, num_warps=4)
    return result


@triton.jit
def _wo_a_reduce_quant(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    v = tl.load(P + split[:, None] * (M * 2048) + row * 2048 + i[None, :])
    y = tl.sum(v, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    sf, inv = ue8m0_scale(amax)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 2048 + i, quant.reshape((256,)))
    col = tile * 8 + tl.arange(0, 8)
    off = (col // 4) * 512 + row * 16 + col % 4
    tl.store(S + off, sf.to(tl.uint8))
    # Zero only padding rows; valid scale bytes have disjoint writers above.
    for z in tl.static_range(triton.cdiv(8192, M * 8 * 256)):
        s = (row * 8 + tile) * 256 + tl.arange(0, 256) + z * (M * 8 * 256)
        sr = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 8192) & (sr >= M))


def _quantize_partial(p):
    m = p.shape[1]
    q = torch.empty((m, 2048), device=p.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(8192, device=p.device, dtype=torch.uint8)
    _wo_a_reduce_quant[(m, 8)](p, q, s, m, num_warps=4)
    return q, s


def wo_a_bf16_small_batch_mxfp8(x: torch.Tensor, weight: torch.Tensor):
    """WO-A with BF16 rounding followed by FlashInfer-compatible MXFP8 quantization."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    return _quantize_partial(partial)
