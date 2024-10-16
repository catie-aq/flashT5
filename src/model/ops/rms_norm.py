# Copyright (c) 2023, Tri Dao.
# Copyright 2024 CATIE. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications to the orignal file
# - support for torch.compile

import triton
import triton.language as tl
import torch
import math

@triton.jit
def _rmsnorm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):

    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < N
    if IS_EVEN_N:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    x_hat = x * rstd
    y = x_hat * w

    # Write output
    if IS_EVEN_N:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rmsnorm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_dy_row,
    stride_dx_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row

    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row

    w = tl.load(W + cols, mask=mask).to(tl.float32)

    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    row_end = min((row_block_id + 1) * rows_per_program, M)

    for row in range(row_start, row_end):
        # Load data to SRAM
        if IS_EVEN_N:
            x = tl.load(X + cols).to(tl.float32)
            dy = tl.load(DY + cols).to(tl.float32)
        else:
            x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = x * rstd
        if not IS_EVEN_N:
            xhat = tl.where(mask, xhat, 0.0)

        wdy = w * dy
        dw += dy * xhat

        c1 = tl.sum(xhat * wdy, axis=0) / N
        dx = (wdy - xhat * c1) * rstd

        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row

        DY += stride_dy_row
        DX += stride_dx_row

    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


# Wrapper for triton kernel for torch.compile - should be unecessary for PyTorch 2.3 ?
torch.library.define("flasht5::rmsnorm_triton_fwd", "(Tensor X, Tensor W, float eps) -> (Tensor, Tensor)")

@torch.library.impl("flasht5::rmsnorm_triton_fwd", "default")
def rmsnorm_triton_fwd(X, weight, eps):

    M, N = X.shape

    assert X.stride(-1) == 1

    assert weight.shape == (N,)
    assert weight.stride(-1) == 1

    # allocate output
    Y = torch.empty_like(X)
    assert Y.stride(-1) == 1

    rstd = torch.empty((M,), dtype=torch.float32, device=X.device)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    assert N <= BLOCK_N

    # heuristics for number of warps
    with torch.cuda.device(X.device.index):
        _rmsnorm_fwd_kernel[(M,)](
            X,
            Y,
            weight,
            rstd,
            X.stride(0),
            Y.stride(0),
            N,
            eps,
            BLOCK_N,
            (N % BLOCK_N == 0)
        )

    return Y, rstd


@torch.library.register_fake("flasht5::rmsnorm_triton_fwd", rmsnorm_triton_fwd)
def rmsnorm_triton_fwd_abstract(X, weight, eps):
    M, N = X.shape

    Y = torch.empty_like(X)
    rstd = torch.empty((M,), dtype=torch.float32, device=X.device)

    return Y, rstd

torch.library.define("flasht5::rmsnorm_triton_bwd", "(Tensor dY, Tensor X, Tensor W, Tensor rstd, float eps) -> (Tensor, Tensor)")

@torch.library.impl("flasht5::rmsnorm_triton_bwd", "default")
def rmsnorm_triton_bwd(
    dy,
    x,
    weight,
    rstd,
    eps
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)

    assert weight.shape == (N,)
    assert weight.stride(-1) == 1

    # allocate output
    dx = torch.empty_like(x)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    assert N <= BLOCK_N

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _rmsnorm_bwd_kernel[grid](
            x,
            weight,
            dy,
            dx,
            _dw,
            rstd,
            x.stride(0),
            dy.stride(0),
            dx.stride(0),
            M,
            N,
            eps,
            rows_per_program,
            BLOCK_N,
            (N % BLOCK_N == 0)
        )
    dw = _dw.sum(0).to(weight.dtype)

    return dx, dw


@torch.library.register_fake("flasht5::rmsnorm_triton_bwd", rmsnorm_triton_bwd)
def rmsnorm_triton_bwd_abstract(dy, x, weight, rstd, eps):

    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty((1, N), dtype=torch.float32, device=weight.device)


    return dx, dw


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps=1e-6):

        X_orig_shape = X.shape
        X = X.reshape(-1, X.shape[-1])

        y, rstd, = torch.ops.flasht5.rmsnorm_triton_fwd(X, W, eps)

        y = y.reshape(X_orig_shape)

        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(X, W, rstd)
        ctx.x_shape_og = X_orig_shape
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, dY):
        X, weight, rstd = ctx.saved_tensors
        dY = dY.reshape(-1, dY.shape[-1])

        assert dY.shape == X.shape

        dx, dw = torch.ops.flasht5.rmsnorm_triton_bwd(
            dY,
            X,
            weight,
            rstd,
            ctx.eps
        )

        return dx.reshape(ctx.x_shape_og), dw, None

def fast_rms_layernorm(X, W, eps):
    out = Fast_RMS_Layernorm.apply(X, W, eps)
    return out
