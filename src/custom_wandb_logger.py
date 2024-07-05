from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # We do not want to log the checkpoints as artifacts due to an error:
        # ReferenceError: weakly-referenced object no longer exists
        pass
