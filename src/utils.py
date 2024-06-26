from typing import Any, Hashable
from print_on_steroids import logger
from torch.optim import AdamW
from transformers.optimization import get_scheduler


def configure_optimizer(
    params: dict, global_rank: int, learning_rate, weight_decay, warmup_period, beta1, beta2, epsilon, lr_schedule, trainer
) -> dict:
    if global_rank == 0:
        logger.info(f"Using lr: {learning_rate}, weight decay: {weight_decay} and warmup steps: {warmup_period}")

    named_parameters = list(params)

    ### Filter out parameters that are not optimized (requires_grad == False)
    optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

    ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_parameters,
        learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,  # You can also tune this
    )

    scheduler_name = lr_schedule
    if scheduler_name == "constant" and warmup_period > 0:
        scheduler_name += "_with_warmup"
    scheduler = get_scheduler(
        scheduler_name,
        optimizer,
        num_warmup_steps=int(warmup_period),
        num_training_steps=trainer.max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


def define_wandb_metrics(extra_metrics: list[str] = [], extra_step_metrics: list[str] = []):
    import wandb

    default_metrics = ["train/loss"]
    default_step_metrics = ["progress/tokens", "progress/samples", "trainer/global_step"]

    # First we need to define the step_metrics as metrics to use a step_metric
    for metric in default_step_metrics + extra_step_metrics:
        wandb.define_metric(metric)

    # Then we define the metrics to be logged
    for metric in default_metrics + extra_metrics:
        for step_metric in default_step_metrics + extra_step_metrics:
            wandb.define_metric(metric, step_metric=step_metric)


##############################
# Template pre-defined utils #
##############################


def find_multiple(n: int, k: int) -> int:
    """From lit-gpt."""
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def wait_for_debugger(port: int = 56786):
    """
    Pauses the program until a remote debugger is attached. Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()


def format_elapsed_time(seconds: float):
    if seconds < 0.001:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds %= 60
        return f"{minutes}:{int(seconds):02d}m"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{hours}:{minutes:02d}:{int(seconds):02d}h"
    else:
        days = int(seconds // 86400)
        seconds %= 86400
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{days}:{hours:02d}:{minutes:02d}:{int(seconds):02d}d"


def pretty_str_from_dict(data: dict, prefix: str = ""):
    """
    Utility function to print a dict of metric key-value pairs to the terminal.

    Returns a pretty string to print to the terminal. Uses some heuristics to prettify:
    - if `time` is in the key, we assume it's a elapsed time in seconds and format it accordingly
    - format all floats to 3 decimal places
    - if a key contains a `/`, we assume it's a path and only print the last part
    """
    print_str = prefix + " " if prefix else ""
    for k, v in data.items():
        if "time" in k and isinstance(v, float):
            v = format_elapsed_time(v)
        elif isinstance(v, float):
            v = f"{v:.3f}"

        if "/" in k:
            k = k.split("/")[-1]

        print_str += f"{k}: {v}, "
    return print_str[:-2]  # Remove trailing ", "


class ddict(dict):
    """Wrapper around the native dict class that allows access via dot syntax and JS-like behavior for KeyErrors."""

    def __getattr__(self, key: Hashable) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: Hashable, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Hashable) -> None:
        del self[key]

    def __dir__(self):
        return self.keys()
