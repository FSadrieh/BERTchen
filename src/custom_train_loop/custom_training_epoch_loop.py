from lightning.pytorch.loops import _TrainingEpochLoop
from lightning.pytorch.utilities.exceptions import SIGTERMException
from lightning.pytorch.loops.fetchers import _DataFetcher
import numpy as np
import torch
from print_on_steroids import logger


class CustomTrainingEpochLoop(_TrainingEpochLoop):
    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        trainer,
        min_steps: int,
        max_steps: int,
        monitor: str,
        min_delta: float,
        patience: int,
        mode: str,
        num_datasets: int,
    ):
        super().__init__(trainer, min_steps=min_steps, max_steps=max_steps)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait = 0
        self.monitor_op = self.mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.num_datasets = num_datasets
        self.dataset_switches = 0

    def run(self, data_fetcher: _DataFetcher) -> None:
        self.reset()
        self.on_run_start(data_fetcher)
        while not self.done:
            try:
                self.advance(data_fetcher)
                if self.on_advance_end(data_fetcher) and self.num_datasets - 1 > self.dataset_switches:
                    self.dataset_switches += 1
                    break
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False

    def switch_dataset(self, current: float) -> bool:
        if self.num_datasets > 1:
            if self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
                logger.info(
                    f"No need to switch dataset, the current {self.monitor}: {current} is better than the previous best {self.monitor}: {self.best_score} by more than {self.min_delta}. Waited for {self.wait} epochs already.",
                    rank0_only=True,
                )
                self.best_score = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.wait = 0
                    return True

        return False

    def on_advance_end(self, data_fetcher) -> bool:
        # -----------------------------------------
        # VALIDATE IF NEEDED
        # -----------------------------------------
        switch_datatset = False
        should_check_val = self._should_check_val_fx(data_fetcher)
        if should_check_val:
            self.trainer.validating = True
            self.val_loop.run()
            switch_datatset = self.switch_dataset(current=self.trainer.callback_metrics[self.monitor].squeeze())
            self.trainer.training = True

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # this is increased once per batch disregarding multiple optimizers on purpose for loggers
            self._batches_that_stepped += 1
        # this will save based on the `batches_that_stepped` value
        self._save_loggers_on_train_batch_end()

        # if training finished, defer exit to the parent. this assumes there will be enough time in between
        # which might not be the case depending on what's in the `*_epoch_end` hooks
        if not self._is_training_done and self.trainer.received_sigterm:
            raise SIGTERMException

        return switch_datatset
