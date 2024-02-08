import torch
import triton

import torch.nn.functional as F
from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func

from src.model.ops.flash_attention_v2_bias import attention

# vary seq length for fixed head and batch=4
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 6, 4096, 64
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        for dtype in [torch.bfloat16, torch.float16]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(7, 14)],
                    line_arg="provider",
                    line_vals=["torch", "triton", "flash", "flash-no-bias"],
                    line_names=["PyTorch", "Triton", "Flash-v2", "Flash-v2-no-bias"],
                    styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")],
                    ylabel="flops",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}-dtype={dtype}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "dtype": dtype,
                        "mode": mode,
                        "causal": causal,
                    },
                ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "torch":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        b = torch.randn((BATCH, H, N_CTX, N_CTX), dtype=dtype, device="cuda", requires_grad=True)

        sm_scale = 1.3
        fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=b, is_causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        b = torch.randn((BATCH, H, N_CTX, N_CTX), dtype=dtype, device="cuda", requires_grad=True)

        sm_scale = 1.3
        fn = lambda: attention(q, k, v, b, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        b = torch.randn((BATCH, H, N_CTX, N_CTX), dtype=dtype, device="cuda", requires_grad=True)

        sm_scale = 1.3
        fn = lambda: flash_attn_func(qkv, attn_bias=b, causal=causal, softmax_scale=sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash-no-bias":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)

        sm_scale = 1.3
        fn = lambda: flash_attn_func(qkv, causal=causal, softmax_scale=sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
