import functools

import torch
import torch._custom_ops
from torch.library import Library, impl
import flash_attn_2_cuda


fa2_lib = Library("fa2", "DEF")

fa2_lib.define('fwd(Tensor q, Tensor k, Tensor v, Tensor out, Tensor alibi_slopes, float dropout_p, float softmax_scale, bool causal, int window_size_left, int window_size_right, Tensor attn_bias, bool return_softmax, Tensor gen_) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)')

@impl(fa2_lib, 'fwd', "CUDA")
def cuda_fa2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attn_bias: torch.Tensor,
    return_softmax: bool,
    gen_: torch.Tensor,
):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if attn_bias is not None:
        attn_bias = attn_bias.contiguous()
        attn_bias = attn_bias.to(q.dtype)

    return flash_attn_2_cuda.fwd(q, k, v, out, alibi_slopes, dropout_p, softmax_scale, causal, window_size_left, window_size_right, attn_bias, return_softmax, None)

@impl(fa2_lib, 'fwd', "Meta")
def meta_fa2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attn_bias: torch.Tensor,
    return_softmax: bool,
    gen_: torch.Tensor
):

    round_multiple = lambda x, m: (x + m - 1) // m * m
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    num_heads = q.shape[2]
    head_dim_og = q.shape[3]
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 128)
    seqlen_q_rounded_8 = round_multiple(seqlen_q, 8)
    seqlen_k_rounded_8 = round_multiple(seqlen_k, 8)
    head_dim = round_multiple(head_dim_og, 8)

    if attn_bias is not None:
        batch_size_bias = attn_bias.shape[0]
        num_heads_bias = attn_bias.shape[1]

    return (torch.empty_strided((batch_size, seqlen_q, num_heads, head_dim_og),
                (head_dim*num_heads*seqlen_q, head_dim*num_heads, head_dim, 1), device=q.device, dtype=q.dtype), # out
        q.new_empty((batch_size, seqlen_q, num_heads, head_dim)), # q_padded
        k.new_empty((batch_size, seqlen_k, num_heads, head_dim)), # k_padded
        v.new_empty((batch_size, seqlen_k, num_heads, head_dim)), # v_padded
        q.new_empty((batch_size, seqlen_q, num_heads, head_dim)), # out_padded
        q.new_empty((batch_size_bias, num_heads_bias, seqlen_q_rounded_8, seqlen_k_rounded_8)) if attn_bias is not None else None, # attn_bias
        q.new_empty((batch_size, num_heads, seqlen_q)), # softmax_lse
        q.new_empty((batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded)) if return_softmax and (dropout_p > 0) else None, # p
        torch.empty((2), dtype=torch.int64, device=q.device) # rng_state
        )

fa2_lib.define('bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor dq, Tensor dk, Tensor dv, Tensor alibi_slopes, float dropout_p, float softmax_scale, bool causal, int window_size_left, int window_size_right, bool deterministic, Tensor attn_bias, bool attn_bias_require_grad, Tensor ds, int seqlen_k_orig, Tensor gen_, Tensor rng_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)')

@impl(fa2_lib, 'bwd', "CUDA")
def cuda_fa2_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    attn_bias: torch.Tensor,
    attn_bias_require_grad: bool,
    ds: torch.Tensor,
    seqlen_k_orig: int,
    gen_: torch.Tensor,
    rng_sate: torch.Tensor
):

    dout, q, k, v, out = dout.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(), out.contiguous()
    if attn_bias is not None:
        attn_bias = attn_bias.contiguous()
        attn_bias = attn_bias.to(q.dtype)

    return flash_attn_2_cuda.bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, dropout_p, softmax_scale, causal, window_size_left, window_size_right, deterministic, attn_bias, attn_bias_require_grad, ds, None, rng_sate)

@impl(fa2_lib, 'bwd', "Meta")
def meta_fftconv_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    attn_bias: torch.Tensor,
    attn_bias_require_grad: bool,
    ds: torch.Tensor,
    seqlen_k_orig: int,
    gen_: torch.Tensor,
    rng_sate: torch.Tensor
):

    round_multiple = lambda x, m: (x + m - 1) // m * m
    batch_size = dout.shape[0]
    seqlen_q = dout.shape[1]
    seqlen_k = k.shape[1]
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    num_heads = dout.shape[2]
    head_dim_og = dout.shape[3]
    head_dim = round_multiple(head_dim_og, 8)
    seqlen_q_round8 = round_multiple(seqlen_q, 8)
    seqlen_k_round8 = round_multiple(seqlen_k_orig, 8)

    if attn_bias is not None:
        batch_size_bias = attn_bias.shape[0]
        num_heads_bias = attn_bias.shape[1]

    return (torch.empty_strided((batch_size, seqlen_q, num_heads, head_dim_og),
                (head_dim*num_heads*seqlen_q, head_dim*num_heads, head_dim, 1), device=q.device, dtype=q.dtype),
        torch.empty_strided((batch_size, seqlen_k_orig, num_heads, head_dim_og),
                (head_dim*num_heads*seqlen_k, head_dim*num_heads, head_dim, 1), device=k.device, dtype=k.dtype),
        torch.empty_strided((batch_size, seqlen_k, num_heads, head_dim_og),
                (head_dim*num_heads*seqlen_k, head_dim*num_heads, head_dim, 1), device=v.device, dtype=v.dtype),
        torch.empty_strided((batch_size_bias, num_heads_bias, seqlen_q, seqlen_k_orig),
                (num_heads_bias*seqlen_q_round8*seqlen_k_round8, seqlen_q_round8*seqlen_k_round8, seqlen_q_round8, 1), device=v.device, dtype=v.dtype)
                if attn_bias_require_grad else None,
        q.new_empty((batch_size, num_heads, seqlen_q_rounded))
        )
