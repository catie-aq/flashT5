import torch
import math

from torch.optim import Optimizer
from torch.optim.optimizer import _default_to_fused_or_foreach
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
from typing import Iterable, Tuple
from torch import nn, Tensor

class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        kahan_sum (`bool`, *optional*, defaults to False):
            Whether to use Kahan summation for updating parameters.
        foreach (`bool`, *optional*, defaults to False):
            Whether to use the foreach implementation.
        correct_bias (`bool`, *optional*, defaults to True):
            Whether to correct bias in Adam.
        use_state_dtype (`torch.dtype`, *optional*, defaults to None):
            The dtype to use for optimizer state. If None, use the default dtype.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        kahan_sum: bool = False,
        foreach: bool = False,
        correct_bias: bool = True,
        use_state_dtype: torch.dtype = None
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        assert not (foreach and use_state_dtype is not None), "foreach is not supported with use_state_dtype"

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=foreach, \
            kahan_sum=kahan_sum, correct_bias=correct_bias, use_state_dtype=use_state_dtype)

        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, steps, kahan_comps = [], [], [], [], [], []

            # Initialization
            for p in group['params']:
                if p.grad is None:
                    continue

                params.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamWScale does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if "kahan_comp" not in state:
                    state['step'] = torch.tensor(0, dtype=torch.int32, device=p.device)

                    if group["use_state_dtype"] in [torch.float16, torch.bfloat16]:
                        state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=group["use_state_dtype"])
                        state['exp_avg_sq'] = torch.zeros_like(p, device=p.device, dtype=group["use_state_dtype"])
                    else:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if group["kahan_sum"] and p.dtype in [torch.float16, torch.bfloat16]:
                        state["kahan_comp"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state["kahan_comp"] = None
                        group["kahan_sum"] = False

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                kahan_comps.append(state["kahan_comp"])
                steps.append(state["step"])

            torch._foreach_add_(steps, 1)

            # AdamW step
            if group["foreach"] and _default_to_fused_or_foreach(params, False, False):
                self._foreach_adamwscaled(params,
                                          grads,
                                          exp_avgs,
                                          exp_avg_sqs,
                                          steps,
                                          kahan_comps,
                                          group["lr"],
                                          group["betas"][0],
                                          group["betas"][1],
                                          group["weight_decay"],
                                          group["eps"],
                                          group["kahan_sum"],
                                          group["correct_bias"])
            else:
                self._adamwscaled(params,
                                  grads,
                                  exp_avgs,
                                  exp_avg_sqs,
                                  steps,
                                  kahan_comps,
                                  group["lr"],
                                  group["betas"][0],
                                  group["betas"][1],
                                  group["weight_decay"],
                                  group["eps"],
                                  group["kahan_sum"],
                                  group["correct_bias"])

        return loss

    def _adamwscaled(self,
                    params: list[Tensor],
                    grads: list[Tensor],
                    exp_avgs: list[Tensor],
                    exp_avg_sqs: list[Tensor],
                    steps: list[Tensor],
                    kahan_comps: list[Tensor],
                    lr: float,
                    beta1: float,
                    beta2: float,
                    weight_decay: float,
                    eps: float,
                    do_kahan_sum: bool,
                    correct_bias: bool):

        for i, p in enumerate(params):

            exp_avg, exp_avg_sq, grad, step, kahan_comp = exp_avgs[i], exp_avg_sqs[i], grads[i], steps[i], kahan_comps[i]

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
            denom = exp_avg_sq.sqrt().add_(eps)

            step_size = lr
            if correct_bias:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            # Adapt Step from Adafactor
            step_size = step_size * max(1e-3, self._rms(p.data))

            if do_kahan_sum:
                # Adam step
                kahan_comp.addcdiv_(exp_avg, denom, value=-step_size)

                # update weights with kahan compensation using dev_grads as temp buffer
                grad.copy_(p)
                p.add_(kahan_comp)

                # save error back to kahan compensation for next iteration
                grad.sub_(p, alpha=1)
                kahan_comp.add_(grad, alpha=1)
            else:
                p.addcdiv_(exp_avg, denom, value=-step_size)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if weight_decay > 0.0:
                p.add_(p, alpha=(-lr * weight_decay))

    def _foreach_adamwscaled(self,
                    params: list[Tensor],
                    grads: list[Tensor],
                    exp_avgs: list[Tensor],
                    exp_avg_sqs: list[Tensor],
                    steps: list[Tensor],
                    kahan_comps: list[Tensor],
                    lr: float,
                    beta1: float,
                    beta2: float,
                    weight_decay: float,
                    eps: float,
                    do_kahan_sum: bool,
                    correct_bias: bool):

        grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, kahan_comps])

        for (_, dtype), ((dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_kahan_comps), _) in grouped_tensors.items():
            # Foreach implementation
            torch._foreach_mul_(dev_exp_avgs, beta1)
            torch._foreach_add_(dev_exp_avgs, dev_grads, alpha=1 - beta1)

            torch._foreach_mul_(dev_exp_avg_sqs, beta2)
            torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, 1 - beta2)

            # Compute denominator
            torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
            torch._foreach_sqrt_(dev_grads)
            torch._foreach_add_(dev_grads, eps)

            step_size = [torch.tensor(lr, dtype=torch.float32, device=p.device) for p in dev_params]

            if correct_bias:
                torch._foreach_mul_(step_size,
                                   [torch.tensor((math.sqrt(1 - beta2 ** steps[i].item()) / (1 - beta1 ** steps[i].item()) ), dtype=torch.float32, device=p.device)
                                        for i, p in enumerate(dev_params)])

            # Adapt step size using RMS of parameters
            rms_p = torch._foreach_norm(dev_params)
            numel = [torch.tensor(math.sqrt(p.numel())) for p in dev_params]
            torch._foreach_div_(rms_p, numel)
            torch._foreach_maximum_(rms_p, 1e-3)

            torch._foreach_mul_(step_size, rms_p)
            torch._foreach_div_(dev_grads, step_size)

            # explicitly delete tensors when not used
            del rms_p
            del numel
            del step_size

            # Update parameters
            if do_kahan_sum:
                # Adam step
                torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, value=-1)

                # update weights with kahan compensation using dev_grads as temp buffer
                torch._foreach_copy_(dev_grads, dev_params)
                torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

                # save error back to kahan compensation for next iteration
                torch._foreach_sub_(dev_grads, dev_params, alpha=1)
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
            else:
                torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, value=-1)

            # Weight decay
            if weight_decay > 0.0:
                torch._foreach_add_(dev_params, dev_params, alpha=-weight_decay * lr)
