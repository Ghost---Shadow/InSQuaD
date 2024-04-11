import wandb


def wandb_safe_log(*args, **kwargs):
    try:
        wandb.log(*args, **kwargs)
    except Exception as e:
        # Dont crash my training
        ...
