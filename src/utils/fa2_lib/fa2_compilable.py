# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_2_cuda as flash_attn_cuda

# isort: on

torch.library.define("fa2::fwd", "(Tensor q, Tensor k, Tensor v, Tensor out, Tensor alibi_slopes, float dropout_p, float softmax_scale, bool causal, int window_size_left, int window_size_right, Tensor attn_bias, bool return_softmax, Tensor gen_) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")

@torch.library.impl("fa2::fwd", "default")
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

    out, q, k, v, out_padded, attn_bias, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(q, k, v, out, alibi_slopes, dropout_p, softmax_scale, causal, window_size_left, window_size_right, attn_bias, return_softmax, None)
    return  out, q, k, v, out_padded, attn_bias, softmax_lse, S_dmask, rng_state

@torch.library.impl_abstract("fa2::fwd", cuda_fa2_fwd)
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

torch.library.define("fa2::bwd", "(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor dq, Tensor dk, Tensor dv, Tensor alibi_slopes, float dropout_p, float softmax_scale, bool causal, int window_size_left, int window_size_right, bool deterministic, Tensor attn_bias, bool attn_bias_require_grad, Tensor ds, int seqlen_k_orig, Tensor gen_, Tensor rng_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

@torch.library.impl("fa2::bwd", "default")
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
    dq, dk, dv, ds, s = flash_attn_cuda.bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, dropout_p, softmax_scale, causal, window_size_left, window_size_right, deterministic, attn_bias, attn_bias_require_grad, ds, None, rng_sate)
    return dq, dk, dv, ds, s

@torch.library.impl_abstract("fa2::bwd", cuda_fa2_bwd)
def meta_fa2_bwd(
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


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_softmax,
        return_ds
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        out, q_padded, k_padded, v_padded, out_padded, attn_bias_padded, softmax_lse, S_dmask, rng_state = torch.ops.fa2.fwd(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            None,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            attn_bias,
            return_softmax and dropout_p > 0,
            None
        )

        ## WORKAROUND a Pytorch bug, should use _padded version of the tensors but this is buggy when passing them directly to save_for_backward
        ## For now, this breaks the backward when headdim is not a multiple of 8 and/or seqlen_q, seqlen_k are not a multiple of 8
        ## TODO: make the padding here instead
        ctx.save_for_backward(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], out, softmax_lse, rng_state, attn_bias, alibi_slopes)
        #ctx.save_for_backward(q_padded, k_padded, v_padded, out_padded, softmax_lse, rng_state, attn_bias_padded, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.deterministic = deterministic
        ctx.bias_requires_grad = True if attn_bias is not None and return_ds else False
        ctx.seqlen_k_orig = qkv.shape[1]

        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state, attn_bias, alibi_slopes = ctx.saved_tensors

        dq, dk, dv, ds, _ = torch.ops.fa2.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            attn_bias,
            ctx.bias_requires_grad,
            None,
            ctx.seqlen_k_orig,
            None,
            rng_state
        )
        dqkv = torch.stack([dq, dk, dv], dim=2)
        return dqkv, None, None, None, None, None, None, None, ds, None, None

class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_softmax,
        return_ds
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, q_padded, k_padded, v_padded, out_padded, attn_bias_padded, softmax_lse, S_dmask, rng_state = torch.ops.fa2.fwd(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            None,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            attn_bias,
            return_softmax and dropout_p > 0,
            None
        )

        ## WORKAROUND a Pytorch bug, should use _padded version of the tensors but this is buggy when passing them directly to save_for_backward
        ## For now, this breaks the backward when headdim is not a multiple of 8 and/or seqlen_q, seqlen_k are not a multiple of 8
        ## TODO: make the padding here instead
        ctx.save_for_backward(q, kv[:, :, 0], kv[:, :, 1], out, softmax_lse, rng_state, attn_bias, alibi_slopes)
        #ctx.save_for_backward(q_padded, k_padded, v_padded, out_padded, softmax_lse, rng_state, attn_bias_padded, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.deterministic = deterministic
        ctx.bias_requires_grad = True if attn_bias is not None and return_ds else False
        ctx.seqlen_k_orig = kv.shape[1]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state, attn_bias, alibi_slopes = ctx.saved_tensors

        dq, dk, dv, ds, _ = torch.ops.fa2.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            attn_bias,
            ctx.bias_requires_grad,
            None,
            ctx.seqlen_k_orig,
            None,
            rng_state
        )
        dkv = torch.stack([dk, dv], dim=2)

        return dq, dkv, None, None, None, None, None, None, None, ds, None, None

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_softmax,
        return_ds
    ):

        batch_size, seqlen_q = q.shape[:2]
        seqlen_k = k.shape[1]

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if attn_bias is not None:
            attn_bias = attn_bias.to(q.dtype)

        out, q_padded, k_padded, v_padded, out_padded, attn_bias_padded, softmax_lse, S_dmask, rng_state = torch.ops.fa2.fwd(
            q,
            k,
            v,
            None,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            attn_bias,
            return_softmax and dropout_p > 0,
            None
        )

        ## WORKAROUND a Pytorch bug, should use _padded version of the tensors but this is buggy when passing them directly to save_for_backward
        ## For now, this breaks the backward when headdim is not a multiple of 8 and/or seqlen_q, seqlen_k are not a multiple of 8
        ## TODO: make the padding here instead
        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state, attn_bias, alibi_slopes)
        #ctx.save_for_backward(q_padded, k_padded, v_padded, out_padded, softmax_lse, rng_state, attn_bias_padded, alibi_slopes)

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.deterministic = deterministic
        ctx.bias_requires_grad = True if attn_bias is not None and return_ds else False
        ctx.seqlen_k_orig = k.shape[1]

        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state, attn_bias, alibi_slopes = ctx.saved_tensors

        dout = dout.contiguous()
        dq, dk, dv, ds, _ = torch.ops.fa2.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            attn_bias,
            ctx.bias_requires_grad,
            None,
            ctx.seqlen_k_orig,
            None,
            rng_state
        )

        return dq, dk, dv, None, None, None, None, None, None, None, ds, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    attn_bias=None,
    return_attn_probs=False,
    return_ds=False
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_attn_probs,
        return_ds
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    attn_bias=None,
    return_attn_probs=False,
    return_ds=False
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_attn_probs,
        return_ds
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    attn_bias=None,
    return_attn_probs=False,
    return_ds=False
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        attn_bias,
        return_attn_probs,
        return_ds,
    )
