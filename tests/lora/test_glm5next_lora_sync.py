# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm.lora.layers.column_parallel_linear import (
    MergedColumnParallelLinearWithLoRA,
)


def _make_wrapper() -> MergedColumnParallelLinearWithLoRA:
    wrapper = MergedColumnParallelLinearWithLoRA.__new__(
        MergedColumnParallelLinearWithLoRA
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.n_slices = 2
    wrapper.output_ids = (1, 1)
    wrapper.output_slices = (2, 3)
    return wrapper


def test_slice_lora_b_accepts_rank_local_weight_sync_tensors() -> None:
    wrapper = _make_wrapper()
    local = [torch.randn(2, 4), torch.randn(3, 4)]

    sliced = wrapper.slice_lora_b(local)

    assert sliced[0] is local[0]
    assert sliced[1] is local[1]


def test_slice_lora_b_still_slices_full_tensors() -> None:
    wrapper = _make_wrapper()
    full = [
        torch.arange(16).reshape(4, 4),
        torch.arange(24).reshape(6, 4),
    ]

    sliced = wrapper.slice_lora_b(full)

    torch.testing.assert_close(sliced[0], full[0][2:4])
    torch.testing.assert_close(sliced[1], full[1][3:6])


def test_set_lora_splits_one_fused_tensor_by_output_size() -> None:
    wrapper = _make_wrapper()
    wrapper.base_layer = SimpleNamespace(output_sizes=(2, 3))
    wrapper.tp_size = 1
    wrapper.lora_a_stacked = (
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
    )
    wrapper.lora_b_stacked = (
        torch.zeros(1, 1, 2, 1),
        torch.zeros(1, 1, 3, 1),
    )
    lora_a = torch.arange(4).reshape(1, 4)
    lora_b = torch.arange(5).reshape(5, 1)

    wrapper.set_lora(0, lora_a=lora_a, lora_b=lora_b)

    torch.testing.assert_close(wrapper.lora_a_stacked[0][0, 0], lora_a)
    torch.testing.assert_close(wrapper.lora_a_stacked[1][0, 0], lora_a)
    torch.testing.assert_close(wrapper.lora_b_stacked[0][0, 0], lora_b[:2])
    torch.testing.assert_close(wrapper.lora_b_stacked[1][0, 0], lora_b[2:])
