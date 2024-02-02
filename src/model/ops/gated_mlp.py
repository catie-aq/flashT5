import torch
import math
import triton
import triton.language as tl

## Activation function from https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_activations.py

_kAlpha = math.sqrt(2.0 / math.pi)

def gelu_ref(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + torch.tanh(_kAlpha * (x + 0.044715 * x * x * x)))

def gelu_grad_ref(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)

# ReLU
@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    return tl.where(x >= 0, x, 0.0)


@triton.jit
def relu_grad(x):
    # ReLU is different from other activations
    # in that it does not require the input to retrospectively compute its gradient
    # here the input is the downstream gradient, and we return the upstream gradient directly
    return tl.where(x >= 0, 1.0, 0.0)

@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def gelu_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit
def gated_matmul_fwd(
    # Pointers to matrices
    out, input, w1, w2,
    act_input_1, act_input_2,
    # Matrix dimensions
    M, N, K,
    stride_om,
    stride_im,
    stride_wn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_GELU: tl.constexpr,
    SAVE_ACTIVATION_INPUTS: tl.constexpr,
    IS_EVEN_MNK: tl.constexpr
):

    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight 1 has shape (K, N)
    - Weight 2 has shape (K, N)
    - Output has shape (M, N)

    """

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of programs ids along the N axis

    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    input_block_ptr = tl.make_block_ptr(
        base=input,
        shape=(M, K),
        strides=(stride_im, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    w1_block_ptr = tl.make_block_ptr(
        base=w1,
        shape=(K, N),
        strides=(1, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    w2_block_ptr = tl.make_block_ptr(
        base=w2,
        shape=(K, N),
        strides=(1, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    # initialize and iteratively update accumulator
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, K, BLOCK_K):

        if IS_EVEN_MNK:
            x = tl.load(input_block_ptr)
            w1_blk = tl.load(w1_block_ptr)
            w2_blk = tl.load(w2_block_ptr)
        else:
            x = tl.load(input_block_ptr, boundary_check=(0, 1))
            w1_blk = tl.load(w1_block_ptr, boundary_check=(0, 1))
            w2_blk = tl.load(w2_block_ptr, boundary_check=(0, 1))

        acc1 += tl.dot(x, w1_blk)
        acc2 += tl.dot(x, w2_blk)

        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_K))
        w1_block_ptr = tl.advance(w1_block_ptr, (BLOCK_K, 0))
        w2_block_ptr = tl.advance(w2_block_ptr, (BLOCK_K, 0))

    if SAVE_ACTIVATION_INPUTS:
        act_in_1_ptrs = tl.make_block_ptr(
            base=act_input_1,
            shape=(M, N),
            strides=(stride_om, 1),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        act_in_2_ptrs = tl.make_block_ptr(
            base=act_input_2,
            shape=(M, N),
            strides=(stride_om, 1),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        if IS_EVEN_MNK:
            tl.store(act_in_1_ptrs, acc1)
            tl.store(act_in_2_ptrs, acc2)
        else:
            tl.store(act_in_1_ptrs, acc1, boundary_check=(0, 1))
            tl.store(act_in_2_ptrs, acc2, boundary_check=(0, 1))

    if USE_GELU:
        acc1 = gelu(acc1)
    else:
        acc1 = relu(acc1)

    # gating
    acc = acc1 * acc2

    # write back result
    out_ptrs = tl.make_block_ptr(
        base=out,
        shape=(M, N),
        strides=(stride_om, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if IS_EVEN_MNK:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, boundary_check=(0, 1))

@triton.jit
def gated_matmul_bwd_ygrad(
    dout,
    y1_grad, y2_grad,
    act_input_1, act_input_2,
    M, N,
    stride_dom,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_GELU: tl.constexpr,
    IS_EVEN_MNK: tl.constexpr):

    """
    Kernel for backward gated MLP

    Ref :
    y2_grad = torch.mul(gelu(x @ w1), dout)
    y1_grad = torch.mul(gelu_grad(x @ w1) * (x @ w2), dout)
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block pointers
    actin_1_block_ptr = tl.make_block_ptr(
        base=act_input_1,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    actin_2_block_ptr = tl.make_block_ptr(
        base=act_input_2,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        base=dout,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if IS_EVEN_MNK:
        dout_blk = tl.load(dout_block_ptr)
        actin_1_blk = tl.load(actin_1_block_ptr)
        actin_2_blk = tl.load(actin_2_block_ptr)
    else:
        dout_blk = tl.load(dout_block_ptr, boundary_check=(0, 1))
        actin_1_blk = tl.load(actin_1_block_ptr, boundary_check=(0, 1))
        actin_2_blk = tl.load(actin_2_block_ptr, boundary_check=(0, 1))

    if USE_GELU:
        actin_act = gelu(actin_1_blk)
        actin_act_grad = gelu_grad(actin_1_blk)
    else:
        actin_act = relu(actin_1_blk)
        actin_act_grad = relu_grad(actin_1_blk)

    actin_act *= dout_blk # y2_grad
    actin_act_grad *= actin_2_blk
    actin_act_grad *= dout_blk # y1_grad

    y1_grad_ptrs = tl.make_block_ptr(
        base=y1_grad,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    y2_grad_ptrs = tl.make_block_ptr(
        base=y2_grad,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if IS_EVEN_MNK:
        tl.store(y1_grad_ptrs, actin_act_grad)
        tl.store(y2_grad_ptrs, actin_act)
    else:
        tl.store(y1_grad_ptrs, actin_act_grad, boundary_check=(0, 1))
        tl.store(y2_grad_ptrs, actin_act, boundary_check=(0, 1))


@triton.jit
def gated_matmul_bwd_input(
    # Pointers to matrices
    w1, w2, # weights inputs
    y1_grad, y2_grad, # partial computation
    din,  # outputs
    # Matrix dimensions
    M, N, K,
    stride_dom, stride_im,
    stride_wn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_EVEN_MNK: tl.constexpr
):

    """
    Kernel for backward gated MLP
    We group along the N axis

    Ref :
    x_grad = torch.matmul(y2_grad, w2.t()) + torch.matmul(y1_grad, w1.t())
    """

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_k = tl.cdiv(K, BLOCK_K)  # number of programs ids along the K axis

    num_pid_in_group = GROUP_M * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_k = (pid % num_pid_in_group) // GROUP_M

    y1_grad_block_ptr = tl.make_block_ptr(
        base=y1_grad,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    y2_grad_block_ptr = tl.make_block_ptr(
        base=y2_grad,
        shape=(M, N),
        strides=(stride_dom, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    w1_block_ptr = tl.make_block_ptr(
        base=w1,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    w2_block_ptr = tl.make_block_ptr(
        base=w2,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    # initialize and iteratively update accumulator
    acc_dx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):

        if IS_EVEN_MNK:
            w1_blk = tl.load(w1_block_ptr)
            w2_blk = tl.load(w2_block_ptr)
            y1_grad_blk = tl.load(y1_grad_block_ptr)
            y2_grad_blk = tl.load(y2_grad_block_ptr)
        else:
            w1_blk = tl.load(w1_block_ptr, boundary_check=(0, 1))
            w2_blk = tl.load(w2_block_ptr, boundary_check=(0, 1))
            y1_grad_blk = tl.load(y1_grad_block_ptr, boundary_check=(0, 1))
            y2_grad_blk = tl.load(y2_grad_block_ptr, boundary_check=(0, 1))

        acc_dx += tl.dot(y2_grad_blk, w2_blk)
        acc_dx += tl.dot(y1_grad_blk, w1_blk)

        w1_block_ptr = tl.advance(w1_block_ptr, (BLOCK_N, 0))
        w2_block_ptr = tl.advance(w2_block_ptr, (BLOCK_N, 0))
        y1_grad_block_ptr = tl.advance(y1_grad_block_ptr, (0, BLOCK_N))
        y2_grad_block_ptr = tl.advance(y2_grad_block_ptr, (0, BLOCK_N))

    # write back result
    dx_ptrs = tl.make_block_ptr(
        base=din,
        shape=(M, K),
        strides=(stride_im, 1),
        offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    if IS_EVEN_MNK:
        tl.store(dx_ptrs, acc_dx)
    else:
        tl.store(dx_ptrs, acc_dx, boundary_check=(0, 1))


@triton.jit
def gated_matmul_bwd_weights(
    # Pointers to matrices
    input,
    y1_grad, y2_grad, # precomputations
    dw1, dw2, # outputs
    # Matrix dimensions
    M, N, K,
    stride_dom, stride_im,
    stride_wn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, GROUP_N: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_EVEN_MNK: tl.constexpr
):

    """
    Kernel for backward gated MLP
    We group along the M axis

    Ref :
    w1_grad = torch.matmul(y1_grad.t(), x)
    w2_grad = torch.matmul(y2_grad.t(), x)
    """

    pid = tl.program_id(0)

    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of program ids along the M axis
    num_pid_k = tl.cdiv(K, BLOCK_K)  # number of programs ids along the K axis

    num_pid_in_group = GROUP_N * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_n = group_id * GROUP_N  # row-id of the first program in the group
    GROUP_N = min(
        num_pid_n - first_pid_n, GROUP_N
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_n = first_pid_n + (pid % GROUP_N)
    pid_k = (pid % num_pid_in_group) // GROUP_N

    # block pointers
    y1_grad_block_ptr = tl.make_block_ptr(
        base=y1_grad,
        shape=(N, M),
        strides=(1, stride_dom),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(0, 1),
    )

    y2_grad_block_ptr = tl.make_block_ptr(
        base=y2_grad,
        shape=(N, M),
        strides=(1, stride_dom),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(0, 1),
    )

    input_block_ptr = tl.make_block_ptr(
        base=input,
        shape=(M, K),
        strides=(stride_im, 1),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # initialize and iteratively update accumulator
    acc_dw1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc_dw2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for i in range(0, M, BLOCK_M):

        if IS_EVEN_MNK:
            y1grad_blk = tl.load(y1_grad_block_ptr)
            y2grad_blk = tl.load(y2_grad_block_ptr)
            x = tl.load(input_block_ptr)
        else:
            y1grad_blk = tl.load(y1_grad_block_ptr, boundary_check=(0, 1))
            y2grad_blk = tl.load(y2_grad_block_ptr, boundary_check=(0, 1))
            x = tl.load(input_block_ptr, boundary_check=(0, 1))

        acc_dw1 += tl.dot(y1grad_blk, x)
        acc_dw2 += tl.dot(y2grad_blk, x)

        y1_grad_block_ptr = tl.advance(y1_grad_block_ptr, (0, BLOCK_M))
        y2_grad_block_ptr = tl.advance(y2_grad_block_ptr, (0, BLOCK_M))
        input_block_ptr = tl.advance(input_block_ptr, (BLOCK_M, 0))

    # write back result
    dw1_ptrs = tl.make_block_ptr(
        base=dw1,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(pid_n * BLOCK_N, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    dw2_ptrs = tl.make_block_ptr(
        base=dw2,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(pid_n * BLOCK_N, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    if IS_EVEN_MNK:
        tl.store(dw1_ptrs, acc_dw1)
        tl.store(dw2_ptrs, acc_dw2)
    else:
        tl.store(dw1_ptrs, acc_dw1, boundary_check=(0, 1))
        tl.store(dw2_ptrs, acc_dw2, boundary_check=(0, 1))


class GatedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, use_gelu=True):

        BLOCK_M = 64
        BLOCK_N = 32
        BLOCK_K = 32
        GROUP_M = 8

        assert x.is_contiguous()
        assert w1.is_contiguous()
        assert w2.is_contiguous()
        assert w1.shape == w2.shape
        assert x.shape[2] == w1.shape[1]
        assert x.shape[2] == w2.shape[1]

        x_ = x if x.ndim == 2 else x.flatten(0, -2)

        M, K = x_.shape
        N, K = w1.shape

        IS_EVEN_MNK = ((M % BLOCK_M) == 0) and ((N % BLOCK_N) == 0) and ((K % BLOCK_K) == 0)

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        act_input_1, act_input_2 = None, None
        if x.requires_grad:
            act_input_1 = torch.empty_like(out)
            act_input_2 = torch.empty_like(out)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        gated_matmul_fwd[grid](
            out,
            x_, w1, w2,
            act_input_1, act_input_2,
            M, N, K,
            out.stride(0), x_.stride(0),
            w1.stride(0),
            BLOCK_M, GROUP_M, BLOCK_N, BLOCK_K,
            use_gelu,
            x.requires_grad,
            IS_EVEN_MNK
        )

        ctx.save_for_backward(x_, w1, w2, act_input_1, act_input_2)
        ctx.use_gelu = use_gelu
        ctx.is_even_nmk = IS_EVEN_MNK
        ctx.x_shape = x.shape

        out = out if x.ndim == 2 else out.reshape(*x.shape[:-1], N)

        return out

    @staticmethod
    def backward(ctx, dout):
        BLOCK_M = 64
        BLOCK_N = 32
        BLOCK_K = 32
        GROUP_M = 8

        x_, w1, w2, act_input_1, act_input_2 = ctx.saved_tensors

        M, K = x_.shape
        N, K = w1.shape

        din = torch.empty_like(x_)
        dw1 = torch.empty_like(w1)
        dw2 = torch.empty_like(w2)

        dout_ = dout if dout.ndim == 2 else dout.flatten(0, -2)

        y1_grad = torch.empty_like(dout_)
        y2_grad = torch.empty_like(dout_)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gated_matmul_bwd_ygrad[grid](
            dout_,
            y1_grad, y2_grad,
            act_input_1, act_input_2,
            M, N,
            dout_.stride(0),
            # Meta-parameters
            BLOCK_M, BLOCK_N,
            ctx.use_gelu,
            ctx.is_even_nmk)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K),)
        gated_matmul_bwd_input[grid](
            w1, w2,
            y1_grad, y2_grad,
            din,
            M, N, K,
            dout_.stride(0), x_.stride(0),
            w1.stride(0),
            BLOCK_M, GROUP_M,
            BLOCK_N, BLOCK_K,
            ctx.is_even_nmk)

        # reorder sizes
        BLOCK_M = 32
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(K, BLOCK_K),)
        gated_matmul_bwd_weights[grid](
            x_,
            y1_grad, y2_grad,
            dw1, dw2,
            M, N, K,
            y1_grad.stride(0), x_.stride(0),
            dw1.stride(0),
            BLOCK_M, GROUP_M,
            BLOCK_N, BLOCK_K,
            ctx.is_even_nmk)

        din = din if len(ctx.x_shape) == 2 else din.reshape(ctx.x_shape)

        return din, dw1, dw2, None

gated_mlp = GatedMLP.apply
