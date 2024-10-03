import torch
from benchmark import Benchmark
from functools import partial
import random
import math

from src.model.modeling_flash_t5 import FlashT5CrossEntropyLoss

@Benchmark.parametrize("batch_size", [16])
@Benchmark.parametrize("sequence_length", [512, 1024])
@Benchmark.parametrize("dtype", [torch.float16, torch.bfloat16])
@Benchmark.parametrize("logits", [32768])
@Benchmark.parametrize("z_loss_factor", [0.0, 1.0])
def test_cross_entropy(batch_size: int, sequence_length: int, dtype: torch.dtype, logits: int, z_loss_factor: float):
    """
    Benchmarks the triton cross entropy loss implementation
    """
    values = []
    shape = (batch_size, sequence_length)
    for _ in range(math.prod(shape)):
        values.append(random.randint(0, 4 - 1))

    labels = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous().cuda()
    input = torch.randn((batch_size, sequence_length, logits), dtype=dtype).cuda().requires_grad_()

    slow_cross_entropy = FlashT5CrossEntropyLoss(z_loss_factor=z_loss_factor, use_triton_crossentropy=False)
    fast_cross_entropy = FlashT5CrossEntropyLoss(z_loss_factor=z_loss_factor, use_triton_crossentropy=True)

    fn_torch = partial(slow_cross_entropy, input, labels)
    fn_triton = partial(fast_cross_entropy, input, labels)

    return {"Torch": fn_torch, "Triton": fn_triton}

print(test_cross_entropy._benchmark.run(mode="fwd_bwd", memory=True, export_graphics=True, key_split="sequence_length", path_graphics="bench_cross_entropy"))
