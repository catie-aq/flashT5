
import triton
import torch
import math
import random

from src.model.ops.cross_entropy_loss import fast_cross_entropy_loss

# vary seq length for fixed head and batch=4
BATCH, N_CTX, LOGITS = 4, 512, 32768
configs = []
for mode in ["fwd", "bwd"]:
    for zloss_factor in [0.0, 1.0, 1.2]:
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(7, 12)],
                        line_arg="provider",
                        line_vals=["torch", "triton"],
                        line_names=["PyTorch", "Triton"],
                        styles=[("red", "-"), ("blue", "-")],
                        ylabel="ms",
                        plot_name=f"fast-cross-entropy-batch{BATCH}-logits{LOGITS}-{mode}-dtype={dtype}-zloss_factor={zloss_factor}",
                        args={
                            "BATCH": BATCH,
                            "LOGITS": LOGITS,
                            "dtype": dtype,
                            "mode": mode,
                            "zloss_factor": zloss_factor
                        },
                    ))

def get_tensors(batch_size, seqlen, logits, dtype):
    values = []
    shape = (batch_size, seqlen)
    for _ in range(math.prod(shape)):
        values.append(random.randint(0, 4 - 1))

    labels = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous().cuda()
    input = torch.randn((batch_size, seqlen, logits), dtype=dtype).cuda().requires_grad_()

    return input, labels

@triton.testing.perf_report(configs)
def bench_cross_entropy(BATCH, N_CTX, LOGITS, mode, provider, dtype=torch.float16, zloss_factor=0.0, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "torch":
        input, labels = get_tensors(BATCH, N_CTX, LOGITS, dtype)

        fn = lambda: fast_cross_entropy_loss(input, labels, z_loss_factor=zloss_factor, use_slow=True)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "triton":
        input, labels = get_tensors(BATCH, N_CTX, LOGITS, dtype)

        fn = lambda: fast_cross_entropy_loss(input, labels, z_loss_factor=zloss_factor, use_slow=False)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


# only works on post-Ampere GPUs right now
bench_cross_entropy.run(save_path=".fast_cross_entropy", print_data=True)
