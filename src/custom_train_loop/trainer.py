from lightning import Trainer


from src.custom_train_loop.custom_training_epoch_loop import CustomTrainingEpochLoop


class CustomTrainer(Trainer):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = "min",
        num_datasets: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fit_loop.epoch_loop = CustomTrainingEpochLoop(
            self,
            min_steps=kwargs.get("min_steps", None),
            max_steps=kwargs.get("max_steps", None),
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            num_datasets=num_datasets,
        )
