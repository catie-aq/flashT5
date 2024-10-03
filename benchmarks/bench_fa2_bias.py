import torch
import torch.nn.functional as F
from benchmark import Benchmark
from functools import partial

from src.model.ops.flash_attention_v2_bias import flash_attention_v2_bias

from flash_attn import flash_attn_func

def flops(batch_size: int, num_heads: int, head_dim: int, sequence_length: int, dtype: torch.dtype, causal: bool, mode="fwd"):
    assert mode in ["fwd", "bwd"]
    f = 4 * batch_size * sequence_length**2 * num_heads * head_dim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f)

@Benchmark.parametrize("batch_size", [16])
@Benchmark.parametrize("num_heads", [12])
@Benchmark.parametrize("head_dim", [64, 128])
@Benchmark.parametrize("sequence_length", [512, 1024])
@Benchmark.parametrize("dtype", [torch.float16, torch.bfloat16])
@Benchmark.parametrize("causal", [True, False])
def test_flash_attention(batch_size: int, num_heads: int, head_dim: int, sequence_length: int, dtype: torch.dtype, causal: bool):
    """
    Benchmarks various flash attention implementations
    """
    # Assuming scaled_dot_product is defined in the same module or imported appropriately
    q = torch.randn((batch_size, num_heads, sequence_length, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((batch_size, num_heads, sequence_length, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((batch_size, num_heads, sequence_length, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    b = torch.randn((1, num_heads, sequence_length, sequence_length), dtype=dtype, device="cuda", requires_grad=True)
    rpe_weights = torch.randn((num_heads, 32), dtype=torch.float32, device="cuda", requires_grad=True)

    sm_scale = 1.3
    fn_spda = partial(F.scaled_dot_product_attention, q, k, v, attn_mask=b, is_causal=causal, scale=sm_scale)
    fn_triton = partial(flash_attention_v2_bias, q, k, v, b, causal=causal, sm_scale=sm_scale)

    q = q.permute((0, 2, 1, 3))
    k = k.permute((0, 2, 1, 3))
    v = v.permute((0, 2, 1, 3))
    fn_flash_attn_rpe = partial(flash_attn_func, q, k, v, softmax_scale=sm_scale, causal=causal, rpe_weights=rpe_weights, rpe_max_distance=256)
    fn_flash_attn = partial(flash_attn_func, q, k, v, softmax_scale=sm_scale, causal=causal)

    return {"SPDA": fn_spda, "Triton": fn_triton, "Flash Attention RPE": fn_flash_attn_rpe, "Flash Attention": fn_flash_attn}

print(test_flash_attention._benchmark.run(mode="fwd_bwd", memory=True, flops=flops, export_graphics=True, key_split="sequence_length"))
