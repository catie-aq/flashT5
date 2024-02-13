import torch
import torch.nn as nn
import triton

from src.model.modeling_flash_t5 import FlashT5LayerNorm

BATCH = 32
configs = []
for mode in ["fwd", "bwd"]:
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for dmodel in [768, 1024]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(7, 14)],
                    line_arg="provider",
                    line_vals=["torch", "triton"],
                    line_names=["PyTorch", "Triton",],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="ms",
                    plot_name=f"layer_norm-batch{BATCH}-d_model{dmodel}-{mode}-dtype={dtype}",
                    args={
                        "BATCH": BATCH,
                        "D_MODEL": dmodel,
                        "dtype": dtype,
                        "mode": mode,
                    },
                ))

@triton.testing.perf_report(configs)
def bench_rms_norm(BATCH, N_CTX, D_MODEL, provider, mode, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]

    warmup = 25
    rep = 100
    if provider == "torch":
        module = FlashT5LayerNorm(D_MODEL, use_triton_layernorm=False).to(device)
        hidden_states = torch.randn((BATCH, N_CTX, D_MODEL), dtype=dtype, device=device, requires_grad=False)
        with torch.autocast(device_type=device, dtype=dtype):
            fn = lambda: module(hidden_states)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "triton":
        module = FlashT5LayerNorm(D_MODEL, use_triton_layernorm=True).to(device)
        hidden_states = torch.randn((BATCH, N_CTX, D_MODEL), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: module(hidden_states)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms

bench_rms_norm.run(save_path=".rms_norm_bench", print_data=False)
