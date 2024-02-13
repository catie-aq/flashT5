import torch
import random
import math

import pytest

from src.model.ops.rms_norm import fast_rms_layernorm
from src.model.modeling_flash_t5 import FlashT5LayerNorm

@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize(
    "batch_size", [4, 6, 7, 8]
)
@pytest.mark.parametrize(
    "seqlen", [64, 128, 256, 512]
)
@pytest.mark.parametrize(
    "model_dim", [768, 1024]
)
def test_layer_norm(batch_size, seqlen, model_dim, dtype):

    input = torch.randn((batch_size, seqlen, model_dim)).cuda().requires_grad_()

    triton_model = FlashT5LayerNorm(model_dim, use_triton_layernorm=True).cuda()
    torch_model = FlashT5LayerNorm(model_dim, use_triton_layernorm=False).cuda()

    with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
        out_ref = torch_model(input)
        out_tri = triton_model(input)
        assert torch.allclose(out_tri, out_ref, atol=1e-2, rtol=0.0)

    out_ref.backward(out_ref)
    input_grad_ref, input.grad = input.grad.clone(), None
    weight_grad_ref = torch_model.weight.grad.clone()

    out_tri.backward(out_tri)
    input_grad_tri, input.grad = input.grad.clone(), None
    weight_grad_tri = triton_model.weight.grad.clone()

    assert torch.allclose(input_grad_ref, input_grad_tri, atol=1e-0, rtol=0.0)
    assert torch.allclose(weight_grad_ref, weight_grad_tri, atol=1e-0, rtol=0.0)
