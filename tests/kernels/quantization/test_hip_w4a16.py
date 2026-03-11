# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.kernels.linear.mixed_precision.hip_w4a16 import (  # noqa: E501
    HipW4A16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not hasattr(
        torch.ops._C, "awq_gemv_hip"
    )  # GEMV op is registered (built and loadable)
    or not current_platform.is_rocm()  # running on ROCm
    or not is_quant_method_supported(
        "awq"
    ),  # important side effect: sets envs.VLLM_USE_TRITON_AWQ on ROCm
    reason="HipW4A16LinearKernel requires ROCm",
)


def _ensure_single_process_model_parallel() -> None:
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file:///tmp/vllm_test_dist",
            backend="gloo",
        )
    if not model_parallel_is_initialized():
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)


def _awq_reference_output(
    w_q_packed: torch.Tensor,
    w_zp_packed: torch.Tensor,
    w_s_data: torch.Tensor,
    x: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    weight_type: ScalarType,
) -> torch.Tensor:
    weights = unpack_quantized_values_into_int32(
        w_q_packed, weight_type, packed_dim=1
    ).to(torch.float32)
    if weights.shape == (n, k):
        weights = weights.t()
    if weights.shape != (k, n):
        raise ValueError(f"Unexpected unpacked weight shape: {tuple(weights.shape)}")

    zeros = unpack_quantized_values_into_int32(
        w_zp_packed, weight_type, packed_dim=0
    ).to(torch.float32)
    if zeros.shape == (n, w_zp_packed.shape[1]):
        zeros = zeros.t()
    if zeros.shape != (k // group_size, n):
        raise ValueError(f"Unexpected zero shape: {tuple(zeros.shape)}")

    scales = w_s_data.to(torch.float32)
    if scales.shape == (n, w_s_data.shape[1]):
        scales = scales.t()
    if scales.shape != (k // group_size, n):
        raise ValueError(f"Unexpected scale shape: {tuple(scales.shape)}")

    zeros_exp = zeros.repeat_interleave(group_size, dim=0)
    scales_exp = scales.repeat_interleave(group_size, dim=0)
    dequant = (weights - zeros_exp) * scales_exp
    return x.to(torch.float32) @ dequant


@pytest.mark.parametrize(
    "pattern",
    ["random", "ones", "last_row", "last_col"],
)
@pytest.mark.parametrize(
    ("group_size", "n", "k"),
    [
        (16, 8, 16),
        (16, 2048, 16 * 5),
        (16, 2048, 16 * 20),
        (32, 8, 32),
        (32, 2048, 32 * 5),
        (32, 2048, 32 * 16),
        (32, 2048, 32 * 80),
        (32, 8192, 32 * 20),
        (64, 2048, 64 * 10),
        (64, 2048, 64 * 20),
        (128, 8, 128),
        (128, 2048, 128 * 1),
        (128, 2048, 128 * 7),
        (128, 2048, 128 * 15),
        (128, 2048, 128 * 16),
        (128, 2048, 128 * 17),
        (128, 2048, 128 * 24),
        (128, 2048, 128 * 27),
        (128, 8192, 128 * 22),
        (128, 8192, 128 * 24),
        (128, 12288, 128 * 20),
        (128, 12288, 128 * 21),
        (128, 16384, 128 * 15),
        (128, 16384, 128 * 22),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])  # TODO: +torch.bfloat16
def test_hip_w4a16_asymmetric_correctness(
    group_size: int,
    n: int,
    k: int,
    pattern: str,
    act_dtype: torch.dtype,
    random_seed: int,
) -> None:
    _ensure_single_process_model_parallel()
    pack_factor = 8
    set_random_seed(random_seed)

    # For bfloat16, use values beyond float16 range (~65504) to verify
    # the kernel doesn't silently narrow to float16.
    scale_multiplier = 1e5 if act_dtype == torch.bfloat16 else 1.0

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err

    if pattern == "random":
        w_q_packed = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (n, k // pack_factor),
            dtype=torch.int32,
            device="cuda",
        )
        w_zp_packed = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (n // pack_factor, k // group_size),
            dtype=torch.int32,
            device="cuda",
        )
        w_s_data = (
            torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
            * scale_multiplier
        )
        x = torch.rand((1, k), dtype=act_dtype, device="cuda") * scale_multiplier
    elif pattern == "last_col":
        # Zeros live in the weights here: only the last output column has
        # non-zero weights. This isolates tail-column handling because the
        # expected output has a single non-zero column regardless of x, so a
        # skipped tail column is immediately visible. Using activation zeros
        # here would still mix non-zero weights into many columns.
        w_unpacked = torch.zeros((n, k), dtype=torch.int32, device="cuda")
        w_unpacked[-1, :] = 1
        w_q_packed = pack_quantized_values_into_int32(
            w_unpacked, scalar_types.uint4, packed_dim=1
        )
        w_zp_packed = torch.zeros(
            (n // pack_factor, k // group_size),
            dtype=torch.int32,
            device="cuda",
        )
        w_s_data = (
            torch.ones((n, k // group_size), dtype=act_dtype, device="cuda")
            * scale_multiplier
        )
        x = torch.rand((1, k), dtype=act_dtype, device="cuda") * scale_multiplier
    elif pattern == "last_row":
        # Zeros live in the activations here: only the last input row
        # contributes. This isolates tail-row handling without constructing a
        # specialized packed weight matrix for every column; if the kernel
        # drops the last K row, the output collapses to zero.
        w_q_packed = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (n, k // pack_factor),
            dtype=torch.int32,
            device="cuda",
        )
        w_zp_packed = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (n // pack_factor, k // group_size),
            dtype=torch.int32,
            device="cuda",
        )
        w_s_data = (
            torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
            * scale_multiplier
        )
        x = torch.zeros((1, k), dtype=act_dtype, device="cuda")
        x[0, -1] = scale_multiplier
    elif pattern == "ones":
        # (q - z) == 1, though we still use randomness as we can
        z_val = torch.randint(0, 15, (1,), dtype=torch.int32, device="cuda").item()
        q_val = z_val + 1
        unpacked = torch.full((1, pack_factor), q_val, dtype=torch.int32, device="cuda")
        packed = int(
            pack_quantized_values_into_int32(
                unpacked, scalar_types.uint4, packed_dim=1
            ).item()
        )
        w_q_packed = torch.full(
            (n, k // pack_factor), packed, dtype=torch.int32, device="cuda"
        )
        unpacked = torch.full((1, pack_factor), z_val, dtype=torch.int32, device="cuda")
        packed = int(
            pack_quantized_values_into_int32(
                unpacked, scalar_types.uint4, packed_dim=1
            ).item()
        )
        w_zp_packed = torch.full(
            (n // pack_factor, k // group_size),
            packed,
            dtype=torch.int32,
            device="cuda",
        )
        w_s_data = (
            torch.ones((n, k // group_size), dtype=act_dtype, device="cuda")
            * scale_multiplier
        )
        x = torch.rand((1, k), dtype=act_dtype, device="cuda") * scale_multiplier
    else:
        raise AssertionError(f"Unknown pattern: {pattern}")

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )
    w_zp = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_zp_packed,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    layer.register_parameter("weight_zero_point", w_zp)

    kernel = HipW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    y = kernel.apply_weights(layer, x)

    y_ref = _awq_reference_output(
        w_q_packed=w_q_packed,
        w_zp_packed=w_zp_packed,
        w_s_data=w_s_data,
        x=x,
        n=n,
        k=k,
        group_size=group_size,
        weight_type=scalar_types.uint4,
    ).to(y.dtype)

    atol = 1e-2 if pattern != "random" else 2e-1
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=atol)


@pytest.mark.parametrize(
    ("group_size", "n", "k"),
    [
        (32, 2048, 32 * 16),
        (32, 2048, 32 * 80),
        (128, 2048, 128 * 7),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])  # TODO: +torch.bfloat16
def test_hip_w4a16_asymmetric_gemm_fallback_correctness(
    group_size: int,
    n: int,
    k: int,
    act_dtype: torch.dtype,
    random_seed: int,
) -> None:
    _ensure_single_process_model_parallel()
    m = 2
    pack_factor = 8
    set_random_seed(random_seed)

    # For bfloat16, use values beyond float16 range (~65504) to verify
    # the kernel doesn't silently narrow to float16.
    scale_multiplier = 1e5 if act_dtype == torch.bfloat16 else 1.0

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n, k // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    w_zp_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n // pack_factor, k // group_size),
        dtype=torch.int32,
        device="cuda",
    )
    w_s_data = (
        torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
        * scale_multiplier
    )
    x = torch.rand((m, k), dtype=act_dtype, device="cuda") * scale_multiplier

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )
    w_zp = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_zp_packed,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    layer.register_parameter("weight_zero_point", w_zp)

    kernel = HipW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    y = kernel.apply_weights(layer, x)

    y_ref = _awq_reference_output(
        w_q_packed=w_q_packed,
        w_zp_packed=w_zp_packed,
        w_s_data=w_s_data,
        x=x,
        n=n,
        k=k,
        group_size=group_size,
        weight_type=scalar_types.uint4,
    ).to(y.dtype)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=2e-1)


@pytest.mark.parametrize(
    ("group_size", "n", "k"),
    [
        (16, 2048, 16 * 20),
        (32, 2048, 32 * 5),
        (32, 2048, 32 * 16),
        (32, 2048, 32 * 80),
        (64, 2048, 64 * 10),
        (128, 2048, 128 * 16),
        (128, 2048, 128 * 7),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])  # TODO: +torch.bfloat16
def test_hip_w4a16_symmetric_correctness(
    group_size: int,
    n: int,
    k: int,
    act_dtype: torch.dtype,
    random_seed: int,
) -> None:
    """Test the symmetric (zero_points=False) path where HipW4A16LinearKernel
    synthesizes an all-zero BasevLLMParameter for zero points."""
    _ensure_single_process_model_parallel()
    pack_factor = 8
    set_random_seed(random_seed)

    # For bfloat16, use values beyond float16 range (~65504) to verify
    # the kernel doesn't silently narrow to float16.
    scale_multiplier = 1e5 if act_dtype == torch.bfloat16 else 1.0

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4b8,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n, k // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    w_s_data = (
        torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
        * scale_multiplier
    )
    x = torch.rand((1, k), dtype=act_dtype, device="cuda") * scale_multiplier

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    # No weight_zero_point registered — kernel should synthesize it

    kernel = HipW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    )
    kernel.process_weights_after_loading(layer)

    # Verify zero points were synthesized
    assert hasattr(layer, "weight_zero_point"), (
        "process_weights_after_loading should synthesize weight_zero_point"
    )

    y = kernel.apply_weights(layer, x)

    # Reference: symmetric path fills zero points with the uint4b8 bias
    bias = scalar_types.uint4b8.bias
    packed_bias = 0
    for i in range(pack_factor):
        packed_bias |= bias << (scalar_types.uint4b8.size_bits * i)
    if packed_bias > 0x7FFFFFFF:
        packed_bias -= 0x100000000
    w_zp_packed = torch.full(
        (n // pack_factor, k // group_size),
        packed_bias,
        dtype=torch.int32,
        device="cuda",
    )
    y_ref = _awq_reference_output(
        w_q_packed=w_q_packed,
        w_zp_packed=w_zp_packed,
        w_s_data=w_s_data,
        x=x,
        n=n,
        k=k,
        group_size=group_size,
        weight_type=scalar_types.uint4b8,
    ).to(y.dtype)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=2e-1)
