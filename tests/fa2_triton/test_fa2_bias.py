import torch
import pytest
from src.model.ops.flash_attention_v2_bias import attention
from src.utils.attn_ref import attn_ref

def max_diff(a, b):
    return (a - b).abs().max().item()

@pytest.mark.parametrize('scale', [1.0])
@pytest.mark.parametrize('B, H, M, N, D', [
    (2, 4, 512, 612, 128),
    (2, 4, 1024, 1045, 64),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_fwd(B, H, M, N, D, causal, dtype, scale):
    q = torch.empty((B, H, M, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    k = torch.empty((B, H, N, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    v = torch.empty((B, H, N, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    b = torch.empty((B, H, M, N), dtype=dtype, device="cuda").normal_(mean=0., std=scale)

    o_ref = attn_ref(q, k, v, b, causal=causal, sm_scale=scale, upcast=True)
    o_torch = attn_ref(q, k, v, b, causal=causal, sm_scale=scale, upcast=False)
    o_hyp = attention(q, k, v, b, causal, scale)

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5


@pytest.mark.parametrize('scale', [1.0])
@pytest.mark.parametrize('B, H, M, N, D', [
    (2, 4, 512, 612, 128),
    (2, 4, 1024, 1045, 64),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16])
def test_attention_bwd(B, H, M, N, D, causal, dtype, scale):
    q = torch.empty((B, H, M, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale).requires_grad_()
    k = torch.empty((B, H, N, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale).requires_grad_()
    v = torch.empty((B, H, N, D), dtype=dtype, device="cuda").normal_(mean=0., std=scale).requires_grad_()
    bias = torch.empty((1, 1, M, N), dtype=dtype, device="cuda").normal_(mean=0., std=scale).requires_grad_()
    do = torch.randn((B, H, M, D), dtype=dtype, device="cuda")

    o_ref = attn_ref(q, k, v, bias, causal=causal, sm_scale=scale, upcast=True)
    o_torch = attn_ref(q, k, v, bias, causal=causal, sm_scale=scale, upcast=False)
    o_hyp = attention(q, k, v, bias, causal=causal, sm_scale=scale)

    gq_ref, gk_ref, gv_ref, gb_ref = torch.autograd.grad(o_ref, (q, k, v, bias), do)
    gq_torch, gk_torch, gv_torch, gb_torch = torch.autograd.grad(o_torch, (q, k, v, bias), do)
    gq_hyp, gk_hyp, gv_hyp, gb_hyp = torch.autograd.grad(o_hyp, (q, k, v, bias), do)

    gq_torch_max_diff = max_diff(gq_torch, gq_ref)
    gk_torch_max_diff = max_diff(gk_torch, gk_ref)
    gv_torch_max_diff = max_diff(gv_torch, gv_ref)
    gb_torch_max_diff = max_diff(gb_torch, gb_ref)

    gq_triton_max_diff = max_diff(gq_hyp, gq_ref)
    gk_triton_max_diff = max_diff(gk_hyp, gk_ref)
    gv_triton_max_diff = max_diff(gv_hyp, gv_ref)
    gb_triton_max_diff = max_diff(gb_hyp, gb_ref)

    #assert o_triton_max_diff < 2 * o_torch_max_diff + 1e-5
    assert gq_triton_max_diff < 2 * gq_torch_max_diff + 1e-5
    assert gk_triton_max_diff < 2 * gk_torch_max_diff + 1e-5
    assert gv_triton_max_diff < 2 * gv_torch_max_diff + 1e-5
    assert gb_triton_max_diff < 2 * gb_torch_max_diff + 1e-5

test_attention_fwd(8, 12, 128, 256, 64, False, torch.float16, scale=1.0)
