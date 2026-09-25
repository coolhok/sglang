"""SM90 fused inverse-RoPE + WO-A against the unfused math."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.wo_a import fused_rope_wo_a_bf16_sm90
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="fused_rope_wo_a_bf16_sm90 requires SM90",
)

LOCAL_HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64
MAX_POS = 4096


def _inputs(tokens: int, padded: bool, position_dtype: torch.dtype):
    torch.manual_seed(1234 + tokens + 100 * padded)
    heads = 64 if padded else LOCAL_HEADS
    backing = torch.randn(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    o = backing[:, :LOCAL_HEADS, :]
    weight = torch.randn(1, 1024, 4096, dtype=torch.bfloat16, device="cuda").mul_(0.05)
    angles = torch.outer(
        torch.arange(MAX_POS, device="cuda", dtype=torch.float32),
        1.0 / 10000.0 ** (torch.arange(32, device="cuda", dtype=torch.float32) / 32),
    )
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).reshape(MAX_POS, ROPE_DIM)
    positions = torch.randint(
        0, MAX_POS, (tokens,), dtype=position_dtype, device="cuda"
    )
    return o, weight, freqs, positions


def _reference(o, weight, freqs, positions):
    rotated = o.float().clone()
    f = freqs[positions.long()]
    cos, sin = f[:, 0::2], f[:, 1::2]
    tail = rotated[..., -ROPE_DIM:]
    real = tail[..., 0::2].clone()
    imag = tail[..., 1::2].clone()
    tail[..., 0::2] = (real * cos[:, None, :] + imag * sin[:, None, :]).bfloat16()
    tail[..., 1::2] = (imag * cos[:, None, :] - real * sin[:, None, :]).bfloat16()
    grouped = rotated.bfloat16().view(o.shape[0], 1, -1)
    return torch.einsum("tgd,grd->tgr", grouped, weight)


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_matches_reference(tokens: int, padded: bool, position_dtype: torch.dtype):
    o, weight, freqs, positions = _inputs(tokens, padded, position_dtype)
    grouped = o.view(tokens, 1, -1)
    actual = fused_rope_wo_a_bf16_sm90(grouped, weight, freqs, positions)
    expected = _reference(o, weight, freqs, positions)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 1e-3, f"relative error {error:.3g}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
