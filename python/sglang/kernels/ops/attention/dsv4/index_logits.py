"""The index logits of the DeepSeek V4.1 ratio-1/2 index layers, by the layout of
the index K they read: the flattened K of a prefill chunk (tile by tile under a
memory budget), the paged pool, or the published blocks of a sparse table."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align

from .candidate_table import CANDIDATE_BLOCK_SIZE


@triton.jit
def _index_score_epilogue_kernel(
    scores,
    weights,
    lengths,
    out,
    width,
    stride_sb,
    stride_sh,
    stride_sn,
    stride_wb,
    stride_wh,
    stride_l,
    stride_ob,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_LENGTH_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    heads = tl.arange(0, BLOCK_H)
    valid = (heads[:, None] < H) & (columns[None, :] < width)
    score = tl.load(
        scores
        + row * stride_sb
        + heads[:, None] * stride_sh
        + columns[None, :] * stride_sn,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weights + row * stride_wb + heads * stride_wh,
        mask=heads < H,
        other=0.0,
    ).to(tl.float32)
    relu = tl.maximum(score, 0.0, propagate_nan=tl.PropagateNan.ALL)
    products = (relu * weight[:, None]).to(tl.bfloat16)
    logits = tl.sum(products.to(tl.float32), axis=0).to(tl.bfloat16)
    if APPLY_LENGTH_MASK:
        length = tl.load(lengths + row * stride_l)
        logits = tl.where(columns < length, logits, -float("inf"))
    tl.store(out + row * stride_ob + columns, logits, mask=columns < width)


def index_score_epilogue(
    scores: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ReLU, per-head weights, reduction, and an optional length mask."""
    assert scores.ndim == 3 and weights.shape == scores.shape[:2]
    assert scores.dtype == weights.dtype == torch.bfloat16
    assert scores.is_cuda and weights.is_cuda and scores.device == weights.device
    rows, heads, width = scores.shape
    if lengths is not None:
        assert lengths.ndim == 1 and lengths.shape[0] == rows
        assert lengths.dtype in (torch.int32, torch.int64)
        assert lengths.is_cuda and lengths.device == scores.device
    out = torch.empty((rows, width), dtype=torch.float32, device=scores.device)
    if rows and width:
        score_elements = rows * width
        # Wider tiles reduce launch and MIO pressure, but a small masked tail can
        # dominate; retain the narrower launch for small or poorly aligned scores.
        use_wide_tile = score_elements >= 262_144 and width % 256 == 0
        block_n = 512 if use_wide_tile else 256
        launch_meta = {"num_warps": 2 if use_wide_tile else 4}
        if use_wide_tile and score_elements >= 1_048_576:
            launch_meta["maxnreg"] = 72
        _index_score_epilogue_kernel[(rows, triton.cdiv(width, block_n))](
            scores,
            weights,
            lengths if lengths is not None else weights,
            out,
            width,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            weights.stride(0),
            weights.stride(1),
            lengths.stride(0) if lengths is not None else 0,
            out.stride(0),
            H=heads,
            BLOCK_H=triton.next_power_of_2(heads),
            BLOCK_N=block_n,
            APPLY_LENGTH_MASK=lengths is not None,
            **launch_meta,
        )
    return out


def flat_index_logits_rows_per_tile(
    rows: int, width: int, *, heads: int, budget_bytes: int
) -> int:
    """Rows per fp32 logits tile within ``budget_bytes``, at the kernel's row
    alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=budget_bytes,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)


def flat_index_logits_tiles(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    context_lengths: list[int],
    budget_bytes: int,
    width_align: int = 4,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """``(rows, logits)`` per row tile, each fp32 tile within ``budget_bytes``:
    ``logits[i, j]`` scores query row ``rows.start + i`` against ``kv[starts + j]``,
    garbage past the row's ``lengths``; the width is ``max(context_lengths)``
    aligned to ``width_align``."""
    from deep_gemm import fp8_fp4_mqa_logits

    rows = q[0].shape[0]
    width = ceil_align(max(context_lengths, default=0), width_align)
    if rows == 0 or width == 0:
        return
    rows_per_chunk = flat_index_logits_rows_per_tile(
        rows, width, heads=q[0].shape[1], budget_bytes=budget_bytes
    )
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        tile_starts = starts[tile]
        yield (
            tile,
            fp8_fp4_mqa_logits(
                (q[0][tile], q[1][tile]),
                kv,
                weights[tile],
                tile_starts,
                tile_starts + lengths[tile],
                False,
                width,
            ),
        )


def deep_gemm_fp4_paged_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
) -> torch.Tensor:
    """DeepGEMM paged fp4 logits; no hadamard, the reference does not apply one."""
    from deep_gemm import fp8_fp4_paged_mqa_logits

    sl = seq_lens.to(torch.int32)
    if sl.dim() == 1:
        sl = sl.unsqueeze(-1)
    return fp8_fp4_paged_mqa_logits(
        q_fp4,
        k_cache,
        weights,
        sl,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        False,
    )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    schedule: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16, ``schedule`` the
    ``build_sparse_indexer_schedule`` of the ``topk_blocks`` published blocks."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        schedule,
        topk_blocks,
        CANDIDATE_BLOCK_SIZE,
    )
