import torch
from benchmark import Benchmark
from functools import partial

from src.model.modeling_flash_t5 import FlashT5LayerNorm

@Benchmark.parametrize("batch_size", [16])
@Benchmark.parametrize("sequence_length", [512, 1024])
@Benchmark.parametrize("d_model", [768])
@Benchmark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_norm(batch_size: int, sequence_length: int, d_model: int, dtype: torch.dtype):
    """
    Benchmarks the triton layer norm implementation
    """
    input = torch.randn((batch_size, sequence_length, d_model), dtype=dtype).cuda().requires_grad_()

    slow_cross_entropy = FlashT5LayerNorm(d_model, use_triton_layernorm=False).cuda()
    fast_cross_entropy = FlashT5LayerNorm(d_model, use_triton_layernorm=True).cuda()

    fn_torch = partial(slow_cross_entropy, input)
    fn_triton = partial(fast_cross_entropy, input)

    return {"Torch": fn_torch, "Triton": fn_triton}

print(test_layer_norm._benchmark.run(mode="fwd_bwd", memory=True, export_graphics=True, key_split="sequence_length", path_graphics="bench_layer_norm"))
