import torch
import random
import math

import pytest

from src.model.ops.cross_entropy_loss import fast_cross_entropy_loss

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
    "logits", [32768, 32128, 32102]
)
def test_bias_attention(batch_size, seqlen, logits, dtype):

    values = []
    shape = (batch_size, seqlen)
    for _ in range(math.prod(shape)):
        values.append(random.randint(0, 4 - 1))

    labels = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous().cuda()
    input = torch.randn((batch_size, seqlen, logits), dtype=dtype).cuda().requires_grad_()

    out_ref, z_loss_ref = fast_cross_entropy_loss(input, labels, z_loss_factor=2.0, use_slow=True)
    out_ref += z_loss_ref
    out_ref.backward(out_ref)
    input_grad_ref, input.grad = input.grad.clone(), None

    out_tri, z_loss_tri = fast_cross_entropy_loss(input, labels, z_loss_factor=2.0, use_slow=False)
    out_tri += z_loss_tri
    out_tri.backward(out_tri)
    input_grad_tri, input.grad = input.grad.clone(), None

    print(torch.allclose(z_loss_tri, z_loss_ref, atol=1e-2, rtol=0.0))
    print(torch.allclose(out_tri, out_ref, atol=1e-2, rtol=0.0))
    print(torch.allclose(input_grad_ref, input_grad_tri, atol=1e-2, rtol=0.0))
