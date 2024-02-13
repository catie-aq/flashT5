import torch
import torch.nn as nn
import triton

from src.model.ops.gated_mlp import gated_mlp

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
        #hidden_states = self.dropout(hidden_states)

        return hidden_states

BATCH, D_MODEL = 32, 768
configs = []
for mode in ["fwd", "bwd"]:
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(7, 14)],
                line_arg="provider",
                line_vals=["torch", "triton"],
                line_names=["PyTorch", "Triton",],
                styles=[("red", "-"), ("blue", "-")],
                ylabel="ms",
                plot_name=f"gated_mlp-batch{BATCH}-d_model{D_MODEL}-{mode}-dtype={dtype}",
                args={
                    "BATCH": BATCH,
                    "D_MODEL": D_MODEL,
                    "dtype": dtype,
                    "mode": mode,
                },
            ))

@triton.testing.perf_report(configs)
def bench_gated_mlp(BATCH, N_CTX, D_MODEL, provider, mode, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    module = FlashT5DenseGatedAct(D_MODEL, 1024).to(device)

    warmup = 25
    rep = 100
    if provider == "torch":
        hidden_states = torch.randn((BATCH, N_CTX, D_MODEL), dtype=dtype, device=device, requires_grad=False)
        with torch.autocast(device_type=device, dtype=dtype):
            fn = lambda: module(hidden_states)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "triton":
        hidden_states = torch.randn((BATCH, N_CTX, D_MODEL), dtype=dtype, device=device, requires_grad=True)
        w1 = module.wi_0.weight.to(dtype)
        w2 = module.wi_0.weight.to(dtype)
        fn = lambda: gated_mlp(hidden_states, w1, w2)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms

bench_gated_mlp.run(save_path=".gated_mlp_bench", print_data=False)
