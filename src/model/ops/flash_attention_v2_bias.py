'''
This work is based on Triton original implementation of flash attention (MIT licenced):
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI

Modifications to the original file:
- Fix backward for non-causal attention
- Addition of attention biases
- Support for batch size of type (1,1,len_q,len_k)

'''

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    B_block_ptr,  #
                    qk_scale,  #
                    start, end, step, #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    MASK: tl.constexpr):

    K_block_ptr = tl.advance(K_block_ptr, (0, start))
    V_block_ptr = tl.advance(V_block_ptr, (start, 0))
    B_block_ptr = tl.advance(B_block_ptr, (0, start))

    # loop over k, v and update accumulator
    for start_n in range(start, end, step):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        k = tl.load(K_block_ptr)
        bias = tl.load(B_block_ptr)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        qk = (qk * qk_scale)
        qk += (bias * 1.44269504)  # 1/log(2)

        if MASK:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(q.dtype), v)
        # update m_i and l_i
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# @triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
#    ],
#    key=['N_CTX'],
# )


@triton.jit
def _attn_fwd(Q, K, V, B, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_bz, stride_bh, stride_bq, stride_bk,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              CAUSAL: tl.constexpr  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B + b_offset,
        shape=(N_CTX, N_CTX),
        strides=(stride_bq, stride_bk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, #
                                        B_block_ptr,  #
                                        qk_scale,  #
                                        0, start_m * BLOCK_M, BLOCK_N,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        offs_m, offs_n,
                                        False #
                                        )
    # stage 2: on-band
    tl.debug_barrier()

    if CAUSAL:
        start = start_m * BLOCK_M
        start = tl.multiple_of(start, BLOCK_M)
        end = (start_m + 1) * BLOCK_M
        step = BLOCK_N
    else:
        start = 0
        end = N_CTX
        step = BLOCK_N

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    B_block_ptr,  #
                                    qk_scale,  #
                                    start, end, step,
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                    offs_m, offs_n, CAUSAL#
                                    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :])
    do = tl.load(DO + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, B, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   stride_bq, stride_bk,
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   BLOCK_DMODEL: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):

    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    bT_ptrs = B + offs_m[None, :] * stride_bq + offs_n[:, None] * stride_bk

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m

    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        bias = tl.load(bT_ptrs)

        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        qkT = tl.dot(k, qT)
        qkT += bias

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(qT.dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += BLOCK_M1
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
        bT_ptrs += BLOCK_M1 * stride_bq

    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V, B, sm_scale, #
                 do, DS, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 stride_bq, stride_bk,
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 BLOCK_DMODEL: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr,
                 RETURN_DS: tl.constexpr,
                 USE_DS_ATOMIC_ADD: tl.constexpr):

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    b_ptrs = B + offs_m[:, None] * stride_bq + offs_n[None, :] * stride_bk
    ds_ptrs = DS + offs_m[:, None] * stride_bq + offs_n[None, :] * stride_bk

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n

    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        bias = tl.load(b_ptrs)

        qk = tl.dot(q, kT)
        qk += bias

        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(q.dtype)

        # save ds
        if RETURN_DS:
            if USE_DS_ATOMIC_ADD:
                tl.atomic_add(ds_ptrs, ds)
            else:
                tl.store(ds_ptrs, ds)

        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += BLOCK_N2
        kT_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tok
        b_ptrs += BLOCK_N2 * stride_bk
        ds_ptrs += BLOCK_N2 * stride_bk

    return dq


@triton.jit
def _attn_bwd(Q, K, V, B, sm_scale,  #
              DO,  #
              DQ, DK, DV, DS, #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d, #
              stride_bz, stride_bh, stride_bq, stride_bk, #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,
              CAUSAL: tl.constexpr,
              RETURN_DS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    off_z = bhid // H
    off_h = bhid % H

    adj = stride_h * off_h.to(tl.int64) + stride_z * off_z.to(tl.int64)
    adj_b = stride_bh * off_h.to(tl.int64) + stride_bz * off_z.to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    B += adj_b
    DS += adj_b

    # load scales
    offs_k = tl.arange(0, BLOCK_DMODEL)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps_per_block = BLOCK_N1 // MASK_BLOCK_M1

    if CAUSAL:
        dk, dv = _attn_bwd_dkdv(dk, dv,  #
                                Q, k, v, B, sm_scale,  #
                                DO,  #
                                M, D,  #
                                stride_tok, stride_d,  #
                                stride_bq, stride_bk,
                                H, N_CTX,  #
                                MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,  #
                                start_n, start_m, num_steps_per_block,  #
                                MASK=True  #
                                )

    if CAUSAL:
        start_m += num_steps_per_block * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m) // BLOCK_M1
    else:
        start_m = 0
        num_steps = N_CTX // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, B, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        stride_bq, stride_bk,
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps_per_block = BLOCK_M2 // MASK_BLOCK_N2
    if CAUSAL:
        dq = _attn_bwd_dq(dq, q, K, V, B, sm_scale,  #
                        do, DS, m, D,  #
                        stride_tok, stride_d,  #
                        stride_bq, stride_bk,  #
                        H, N_CTX,  #
                        BLOCK_M2, MASK_BLOCK_N2, BLOCK_DMODEL,  #
                        start_m, end_n - num_steps_per_block * MASK_BLOCK_N2, num_steps_per_block,  #
                        MASK=CAUSAL, #
                        RETURN_DS=RETURN_DS,  #
                        USE_DS_ATOMIC_ADD=((stride_bz == 0) or (stride_bh == 0))
                        )

    # stage 2
    if CAUSAL:
        end_n -= num_steps_per_block * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
    else:
        end_n = N_CTX
        num_steps = end_n // BLOCK_N2

    dq = _attn_bwd_dq(dq, q, K, V, B, sm_scale,  #
                      do, DS, m, D,  #
                      stride_tok, stride_d,  #
                      stride_bq, stride_bk,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, BLOCK_DMODEL,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False,  #
                      RETURN_DS=RETURN_DS,  #
                      USE_DS_ATOMIC_ADD=((stride_bz == 0) or (stride_bh == 0))
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, b, causal, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        # Trick to support shape such as (1, 1, seqlen_q, seqlen_k)
        bias_batch_stride = b.stride(0)
        bias_heads_stride = b.stride(1)
        if (b.shape[0] != q.shape[0]) and (b.shape[0] == 1):
            bias_batch_stride = 0
        if (b.shape[1] != q.shape[1]) and (b.shape[1] == 1):
            bias_heads_stride = 0

        o = torch.empty_like(q)

        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32

        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4

        # Tuning for H100
        if torch.cuda.get_device_capability()[0] == 9:
            num_warps = 8
            num_stages = 7 if Lk >= 64 else 3

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            q, k, v, b, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            bias_batch_stride, bias_heads_stride, b.stride(2), b.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            num_warps=num_warps,  #
            num_stages=num_stages,  #
            CAUSAL=causal
        )

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.bias_batch_stride = bias_batch_stride
        ctx.bias_heads_stride = bias_heads_stride
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, b, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        ds = torch.empty_like(b)

        if (ctx.bias_batch_stride == 0) or (ctx.bias_heads_stride == 0):
            ds = ds.zero_()

        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0

        # prescale matrices
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        arg_b = b * RCP_LN2

        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)

        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, D_HEAD=ctx.BLOCK_DMODEL  #
        )

        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

        _attn_bwd[grid](
            q, arg_k, v, arg_b, ctx.sm_scale, do, dq, dk, dv, ds,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            ctx.bias_batch_stride, ctx.bias_heads_stride, b.stride(2), b.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,
            RETURN_DS=b.requires_grad
        )

        return dq, dk, dv, ds, None, None


attention = _attention.apply
