from lightning.pytorch.callbacks import EarlyStopping
from typing import Optional


class SwitchDatasetCallback(EarlyStopping):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        if should_stop:
            # If the stopping condition is met, we switch the dataset, if we have not reached the final dataset
            if trainer.datamodule.iterator_idx < trainer.datamodule.use_n_training_datasets:
                trainer.fit_loop._last_train_dl_reload_epoch = -trainer.reload_dataloaders_every_n_epochs
                trainer.fit_loop.setup_data()
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
