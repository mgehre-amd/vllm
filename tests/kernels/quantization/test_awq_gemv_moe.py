# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import awq_pack
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not hasattr(torch.ops, "_C")
    or not hasattr(torch.ops._C, "awq_gemv_moe_hip"),
    reason="awq_gemv_moe_hip requires ROCm",
)

NUM_BITS = 4


def _make_awq_moe_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic AWQ-format MoE weights.

    Returns (qweight, scales, qzeros, w_ref):
      qweight: [E, K, N/8] int32 (AWQ packed)
      scales:  [E, K/G, N] fp16
      qzeros:  [E, K/G, N/8] int32 (AWQ packed)
      w_ref:   [E, K, N] fp16 (dequantized reference)
    """
    num_groups = K // group_size
    all_qw, all_scales, all_qz, all_ref = [], [], [], []

    for _ in range(E):
        w_fp = torch.randn(K, N, device=device, dtype=torch.float16) / 10.0
        w_grouped = w_fp.reshape(num_groups, group_size, N)
        abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scales = abs_max / 7.0

        w_q = torch.round(w_grouped / scales).clamp(-7, 7).int() + 8
        w_q = w_q.reshape(K, N)

        zero_point = 8
        z_q = torch.full((num_groups, N), zero_point, dtype=torch.int32, device=device)

        w_ref = (
            ((w_q.float() - zero_point).reshape(num_groups, group_size, N) * scales)
            .reshape(K, N)
            .half()
        )

        qw_packed = awq_pack(w_q, NUM_BITS, K, N)
        qz_packed = awq_pack(z_q, NUM_BITS, num_groups, N)

        all_qw.append(qw_packed)
        all_scales.append(scales.squeeze(1).half())
        all_qz.append(qz_packed)
        all_ref.append(w_ref)

    return (
        torch.stack(all_qw),
        torch.stack(all_scales),
        torch.stack(all_qz),
        torch.stack(all_ref),
    )


def _reference_moe_gemv(
    activation: torch.Tensor,
    w_ref: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    """Reference MoE GEMV using dequantized fp16 weights."""
    num_tokens = activation.shape[0]
    N = w_ref.shape[2]
    num_slots = sorted_token_ids.shape[0]
    num_valid = num_tokens * top_k
    output = torch.zeros(num_slots, N, dtype=torch.float16, device=activation.device)

    for slot_idx in range(num_slots):
        token_id = sorted_token_ids[slot_idx].item()
        if token_id >= num_valid:
            continue
        expert_id = expert_ids[slot_idx].item()
        act_row = token_id // top_k
        result = activation[act_row : act_row + 1].float() @ w_ref[expert_id].float()
        if mul_routed_weight:
            result = result * topk_weights[token_id].item()
        output[token_id] = result.half().squeeze(0)

    return output


@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("n,k", [(256, 256), (512, 256), (256, 512)])
@pytest.mark.parametrize("e,topk", [(4, 2), (8, 2)])
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_awq_gemv_moe_hip(
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    mul_routed_weight: bool,
    random_seed: int,
):
    """Test awq_gemv_moe_hip kernel directly with synthetic AWQ weights.

    This test focuses on MoE-specific routing logic (expert stride offsets,
    sorted_token_ids/expert_ids indexing, mul_routed_weight, top_k handling).
    The underlying GEMV compute uses the same awq_gemv_kernel_splitk template
    as the non-MoE path, so edge-case input patterns (ones, last_row,
    last_col, etc.) are covered by test_hip_w4a16.py and not repeated here.
    """
    from vllm._custom_ops import awq_gemv_moe_hip
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    set_random_seed(random_seed)
    device = torch.device("cuda")
    num_tokens = 1

    qweight, scales, qzeros, w_ref = _make_awq_moe_weights(e, k, n, group_size, device)

    activation = torch.randn(num_tokens, k, device=device, dtype=torch.float16) / 10
    scores = torch.randn(num_tokens, e, device=device, dtype=torch.float16)

    from vllm.model_executor.layers.fused_moe import fused_topk

    topk_weights, topk_ids, _ = fused_topk(activation, scores, topk, False)

    sorted_token_ids, expert_ids, _ = moe_align_block_size(
        topk_ids, 1, e, None, ignore_invalid_experts=True
    )

    num_slots = sorted_token_ids.size(0)
    output = torch.zeros(num_slots, n, dtype=torch.float16, device=device)

    awq_gemv_moe_hip(
        activation,
        qweight,
        scales,
        qzeros,
        output,
        sorted_token_ids,
        expert_ids,
        topk_weights.view(-1),
        topk,
        mul_routed_weight,
        1,
    )

    ref_output = _reference_moe_gemv(
        activation,
        w_ref,
        sorted_token_ids,
        expert_ids,
        topk_weights.view(-1),
        topk,
        mul_routed_weight,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=0)
