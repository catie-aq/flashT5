# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
# - add weights gradients
# - remove the mask if size is a power of 2
# - support for torch.compile

import triton
import triton.language as tl
import torch


MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
    IS_EVEN_X: tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if IS_EVEN_X:
        X_row = tl.load(X + col_offsets).to(tl.float32)
        W_row = tl.load(W + col_offsets)
    else:
        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row

    if IS_EVEN_X:
        tl.store(Y + col_offsets, output)
    else:
        tl.store(Y + col_offsets, output, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,   W_row_stride,
    r,   r_row_stride,
    dW, dW_row_stride,
    dX, dX_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
    IS_EVEN_X: tl.constexpr
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride
    dW += row_idx * dW_row_stride
    dX += row_idx * dX_row_stride

    if IS_EVEN_X:
        dY_row = tl.load(dY + col_offsets).to(tl.float32)
        X_row  = tl.load(X  + col_offsets).to(tl.float32)
        W_row  = tl.load(W  + col_offsets).to(tl.float32)
    else:
        dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
        X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var
    dW_row = dY_row * normed

    dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)

    if IS_EVEN_X:
        tl.store(dW + col_offsets, dW_row)
        tl.store(dX + col_offsets, output)
    else:
        tl.store(dW + col_offsets, dW_row, mask=mask)
        tl.store(dX + col_offsets, output, mask=mask)


# Wrapper for triton kernel for torch.compile - should be unecessary for PyTorch 2.3 ?
torch.library.define("flasht5::rmsnorm_triton_fwd", "(Tensor X, Tensor W, float eps, int n_cols, int n_rows, int BLOCK_SIZE, int num_warps) -> (Tensor, Tensor)")

@torch.library.impl("flasht5::rmsnorm_triton_fwd", "default")
def rmsnorm_triton_fwd(X, W, eps, n_cols, n_rows, BLOCK_SIZE, num_warps):
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda")
    r = torch.empty(n_rows, dtype=torch.float32, device="cuda")

    _rms_layernorm_forward[(n_rows,)](
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        r, r.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_EVEN_X=((n_cols % BLOCK_SIZE) == 0),
        num_warps=num_warps
    )

    return Y, r


@torch.library.impl_abstract("flasht5::rmsnorm_triton_fwd", rmsnorm_triton_fwd)
def rmsnorm_triton_fwd_abstract(X, W, eps, n_cols, n_rows, BLOCK_SIZE, num_warps):
    Y = X.new_empty((n_rows, n_cols))
    r = X.new_empty((n_rows))
    return Y, r

torch.library.define("flasht5::rmsnorm_triton_bwd", "(Tensor dY, Tensor r, Tensor X, Tensor W, float eps, int n_cols, int n_rows, int BLOCK_SIZE, int num_warps) -> (Tensor, Tensor)")

@torch.library.impl("flasht5::rmsnorm_triton_bwd", "default")
def rmsnorm_triton_bwd(dY, r, X, W, eps, n_cols, n_rows, BLOCK_SIZE, num_warps):

    dX = torch.empty_like(dY)
    dW = torch.empty_like(dY)

    _rms_layernorm_backward[(n_rows,)](
        dY, dY.stride(0),
        X,  X.stride(0),
        W,  1,
        r,  1,
        dW, dW.stride(0),
        dX, dX.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_EVEN_X=((n_cols % BLOCK_SIZE) == 0),
        num_warps=num_warps,
    )

    return dX, dW


@torch.library.impl_abstract("flasht5::rmsnorm_triton_bwd", rmsnorm_triton_bwd)
def rmsnorm_triton_bwd_abstract(dY, r, X, W, eps, n_cols, n_rows, BLOCK_SIZE, num_warps):
    return torch.empty_like(dY), torch.empty_like(dY)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y, r = torch.ops.flasht5.rmsnorm_triton_fwd(X, W, eps, n_cols, n_rows, BLOCK_SIZE, num_warps)

        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dX = torch.empty_like(dY)
        dW = torch.empty_like(dY)

        dW, dX = torch.ops.flasht5.rmsnorm_triton_bwd(dY, r, X, W, ctx.eps, n_cols, n_rows, ctx.BLOCK_SIZE, ctx.num_warps)

        dX = dX.view(*shape)
        return dX, dW.sum(0), None

def fast_rms_layernorm(X, W, eps):
    out = Fast_RMS_Layernorm.apply(X, W, eps)
    return out
