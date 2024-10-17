from src.utils.adamw_scaled import AdamWScale

from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR
)

def create_optimizer(model, lr, betas, eps, weight_decay, foreach=False, kahan_sum=False, use_state_dtype=None):

    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamWScale(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        kahan_sum=kahan_sum,
        foreach=foreach,
        use_state_dtype=use_state_dtype
    )

    return optimizer

def create_cosine_scheduler(warmup_steps, warmup_ratio, num_training_steps, optimizer):

    warmup_steps = warmup_steps if warmup_steps != 0 else int(num_training_steps * warmup_ratio)

    if warmup_steps == 0:
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-5,
        )
    else:
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-5,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[warmup_steps]
        )

    return lr_scheduler

def create_wsd_scheduler(warmup_steps, warmup_ratio, num_training_steps, optimizer):

    warmup_steps = warmup_steps if warmup_steps != 0 else int(num_training_steps * warmup_ratio)

    scheduler1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=warmup_steps,
        last_epoch=-1,
    )

    scheduler2 = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=num_training_steps - 2*warmup_steps,
        last_epoch=-1,
    )

    scheduler3 = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.5,
        total_iters=warmup_steps,
        last_epoch=-1,
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[warmup_steps, num_training_steps - warmup_steps]
    )

    return lr_scheduler
