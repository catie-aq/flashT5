import torch
import torch.nn as nn
import random
import math

import pytest

from src.model.ops.gated_mlp import gated_mlp, gelu_torch, gelu_grad_torch

class FlashT5DenseGatedAct(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.0):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_gelu = self.wi_0(hidden_states)
        hidden_gelu = torch.nn.functional.gelu(hidden_gelu)
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        return hidden_states

@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16]
)
@pytest.mark.parametrize(
    "batch_size", [4, 6, 7, 8]
)
@pytest.mark.parametrize(
    "seqlen", [64, 128, 256, 512]
)
@pytest.mark.parametrize(
    "model_dim", [768, 769, 1024]
)
@pytest.mark.parametrize(
    "d_ff", [1024]
)
def test_gated_mlp(batch_size, seqlen, model_dim, d_ff, dtype):

    input = torch.randn((batch_size, seqlen, model_dim)).cuda().requires_grad_()
    torch_model = FlashT5DenseGatedAct(model_dim, d_ff).cuda()

    with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
        out_ref = torch_model(input)
        out_triton = gated_mlp(input, torch_model.wi_0.weight, torch_model.wi_1.weight)

        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=0)

        '''
        dout = torch.randn_like(out_ref)
        out_ref.backward(dout)

        ref_di, input.grad = input.grad.clone(), None
        ref_dw1, torch_model.wi_0.weight.grad = torch_model.wi_0.weight.grad.clone(), None
        ref_dw2, torch_model.wi_1.weight.grad = torch_model.wi_1.weight.grad.clone(), None

        out_triton.backward(dout)
        triton_di, input.grad = input.grad.clone(), None
        triton_dw1, torch_model.wi_0.weight.grad = torch_model.wi_0.weight.grad.clone(), None
        triton_dw2, torch_model.wi_1.weight.grad = torch_model.wi_1.weight.grad.clone(), None

        assert torch.allclose(ref_di, triton_di, atol=1e-1, rtol=0)
        assert torch.allclose(ref_dw1, triton_dw1, atol=1e-1, rtol=0)
        assert torch.allclose(ref_dw2, triton_dw2, atol=1e-1, rtol=0)
        '''
