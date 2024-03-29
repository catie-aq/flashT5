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
# Modification to the original version from Unsloth:
# - return the z-loss
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
def _cross_entropy_forward(logits_ptr, logits_row_stride,
                           loss_ptr,
                           lse_ptr,
                           labels_ptr,
                           n_cols,
                           BLOCK_SIZE: tl.constexpr,
                           IS_EVEN: tl.constexpr):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr   += row_idx
    lse_ptr    += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # TODO: Fixup int32 locations to int64
    label_idx = tl.load(labels_ptr).to(tl.int32)
    if IS_EVEN:
        logits = tl.load(logits_ptr + col_offsets).to(tl.float32)
    else:
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    max_logits = tl.max(logits, 0)

    # Maximum stops overflow
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr, lse)

    if label_idx != -100:
        logits_label = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = lse - logits_label
    else:
        loss = 0.0

    tl.store(loss_ptr, loss)

@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride,
                            dinputs_ptr, dinputs_row_stride,
                            dloss_ptr,  dloss_row_stride,
                            dzloss_ptr, dzloss_row_stride,
                            lse_ptr,
                            labels_ptr,
                            n_cols,
                            BLOCK_SIZE: tl.constexpr,
                            USE_Z_LOSS: tl.constexpr,
                            IS_EVEN: tl.constexpr):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """

    row_idx = tl.program_id(0)

    logits_ptr += row_idx * logits_row_stride
    dinputs_ptr += row_idx * dinputs_row_stride
    dloss_ptr  += row_idx *  dloss_row_stride
    dzloss_ptr  += row_idx *  dzloss_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    # TODO: Fixup int32 locations to int64
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
        dzloss = tl.load(dzloss_ptr)
    else:
        dloss = 0.0
        dzloss = 0.0

    if IS_EVEN:
        logits = tl.load(logits_ptr + col_offsets).to(tl.float32)
    else:
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)

    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    din = dloss * probs

    # Z_loss
    if USE_Z_LOSS:
        if label_idx != -100:
            dzloss = tl.load(dzloss_ptr)
        else:
            dzloss = 0.0

        row_minus_max = logits
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        din += softmax_output * dzloss

    if IS_EVEN:
        tl.store(dinputs_ptr + col_offsets, din)
    else:
        tl.store(dinputs_ptr + col_offsets, din, mask=mask)


# Wrapper for triton kernel for torch.compile - should be unecessary for PyTorch 2.3 ?
torch.library.define("flasht5::cross_entropy_triton_fwd", "(Tensor logits, Tensor labels, int n_cols, int n_rows, int BLOCK_SIZE, int num_warps) -> (Tensor, Tensor)")

@torch.library.impl("flasht5::cross_entropy_triton_fwd", "default")
def cross_entropy_triton_fwd(logits, labels, n_cols, n_rows, BLOCK_SIZE, num_warps):
    losses    = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    logsumexp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

    _cross_entropy_forward[(n_rows,)](
        logits, logits.stride(0),
        losses,
        logsumexp,
        labels,
        n_cols,
        BLOCK_SIZE = BLOCK_SIZE,
        IS_EVEN=((n_cols % BLOCK_SIZE) == 0),
        num_warps  = num_warps,
    )

    return losses, logsumexp


@torch.library.impl_abstract("flasht5::cross_entropy_triton_fwd", cross_entropy_triton_fwd)
def cross_entropy_triton_fwd_abstract(logits, labels, n_cols, n_rows, BLOCK_SIZE, num_warps):
    losses    = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    logsumexp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

    return losses, logsumexp

torch.library.define("flasht5::cross_entropy_triton_bwd", "(Tensor dlosses, Tensor dlogsumexp, Tensor logits, Tensor logsumexp, Tensor labels, float z_loss_factor, int n_cols, int n_rows, int BLOCK_SIZE, int num_warps) -> Tensor")

@torch.library.impl("flasht5::cross_entropy_triton_bwd", "default")
def cross_entropy_triton_bwd(dlosses, dlogsumexp, logits, logsumexp, labels, z_loss_factor, n_cols, n_rows, BLOCK_SIZE, num_warps):

    dinputs = torch.empty_like(logits)

    _cross_entropy_backward[(n_rows,)](
        logits,   logits.stride(0),
        dinputs, dinputs.stride(0),
        dlosses, dlosses.stride(0),
        dlogsumexp, dlogsumexp.stride(0),
        logsumexp,
        labels,
        n_cols,
        BLOCK_SIZE = BLOCK_SIZE,
        USE_Z_LOSS = (z_loss_factor != 0.0),
        IS_EVEN=((n_cols % BLOCK_SIZE) == 0),
        num_warps  = num_warps,
    )

    return dinputs


@torch.library.impl_abstract("flasht5::cross_entropy_triton_bwd", cross_entropy_triton_bwd)
def cross_entropy_triton_bwd_abstract(dlosses, dlogsumexp, logits, logsumexp, labels, z_loss_factor, n_cols, n_rows, BLOCK_SIZE, num_warps):
    return torch.empty_like(logits)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, z_loss_factor):
        n_rows, n_cols = logits.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        losses, logsumexp = torch.ops.flasht5.cross_entropy_triton_fwd(
            logits,
            labels,
            n_cols,
            n_rows,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps
        )

        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.z_loss_factor = z_loss_factor
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses, logsumexp

    @staticmethod
    def backward(ctx, dlosses, dlogsumexp):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape

        dinputs = torch.ops.flasht5.cross_entropy_triton_bwd(
            dlosses,
            dlogsumexp,
            logits,
            logsumexp,
            labels,
            ctx.z_loss_factor,
            n_cols,
            n_rows,
            ctx.BLOCK_SIZE,
            ctx.num_warps
        )
        return dinputs, None, None

def fast_cross_entropy_loss(logits, labels, z_loss_factor=0.0):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))
    assert (d <= MAX_FUSED_SIZE)

    loss, lse = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        z_loss_factor
    )

    n_items = torch.count_nonzero(labels != -100)

    return loss.sum() / n_items, (z_loss_factor * torch.square(lse).sum()) / n_items
