import torch
import random
import math

import pytest

from src.model.modeling_flash_t5 import FlashT5CrossEntropyLoss

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
@pytest.mark.parametrize(
    "z_loss_factor", [0.0, 1.0, 2.0]
)
@pytest.mark.parametrize(
    "label_smoothing", [0.0, 0.1]
)
def test_cross_entropy(batch_size, seqlen, logits, dtype, z_loss_factor, label_smoothing):

    values = []
    shape = (batch_size, seqlen)
    for _ in range(math.prod(shape)):
        values.append(random.randint(0, 4 - 1))

    slow_cross_entropy = FlashT5CrossEntropyLoss(z_loss_factor=z_loss_factor, label_smoothing=label_smoothing, use_triton_crossentropy=False)
    fast_cross_entropy = FlashT5CrossEntropyLoss(z_loss_factor=z_loss_factor, label_smoothing=label_smoothing, use_triton_crossentropy=True)

    labels = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous().cuda()
    input = torch.randn((batch_size, seqlen, logits), dtype=dtype).cuda().requires_grad_()

    out_ref = slow_cross_entropy(input, labels)
    out_ref.backward(out_ref)
    input_grad_ref, input.grad = input.grad.clone(), None

    out_tri = fast_cross_entropy(input, labels)
    out_tri.backward(out_tri)
    input_grad_tri, input.grad = input.grad.clone(), None

    assert torch.allclose(out_tri, out_ref, atol=1e-2, rtol=0.0)
    assert torch.allclose(input_grad_ref, input_grad_tri, atol=1e-2, rtol=0.0)
