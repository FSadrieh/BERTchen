import os
from typing import TYPE_CHECKING

import datasets
import lightning as L
from print_on_steroids import logger
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding, DataCollatorForLanguageModeling

from dlib.frameworks.pytorch import get_rank
from src.custom_data_collator import QADataCollator

if TYPE_CHECKING:
    from train import TrainingArgs


class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        tokenizer: PreTrainedTokenizerFast,
        train_files: list[str],
        val_files: list[str],
    ):
        super().__init__()
        self.args = training_args
        self.data_dir = training_args.data_dir
        self.train_files = [str(self.data_dir / train_files[i]) for i in range(self.args.use_n_train_datasets)]
        self.val_files = [str(self.data_dir / val_files[i]) for i in range(self.args.use_n_val_datasets)]

        logger.debug(f"Train file path: {self.train_files[0]} val file path: {self.val_files[0]}")

        self.local_rank = get_rank()

        self.tokenizer = tokenizer
        self.iterator_idx = 0
        self.use_n_train_datasets = self.args.use_n_train_datasets
        self.use_n_val_datasets = self.args.use_n_val_datasets

    def prepare_data(self) -> None:
        for file in self.train_files + self.val_files:
            if not os.path.exists(file):
                logger.info(f"Could not find processed dataset: {file}, please create it via data download")

    def setup(self, stage):
        logger.info(f"Loading cached processed dataset from {self.data_dir}...", rank0_only=False)

        self.train_datasets = []
        for i in range(self.use_n_train_datasets):
            train_dataset = datasets.load_dataset(
                "json",
                data_files={"train": self.train_files[i]},
                name=str(self.data_dir).replace("/", "_"),
                num_proc=self.args.preprocessing_workers,
            )
            self.train_datasets.append(train_dataset["train"])

        self.val_datasets = []
        for i in range(self.use_n_val_datasets):
            val_dataset = datasets.load_dataset(
                "json",
                data_files={"val": self.val_files[i]},
                name=str(self.data_dir).replace("/", "_"),
                num_proc=self.args.preprocessing_workers,
            )
            self.val_datasets.append(val_dataset["val"])

        self.train_dataset = self.train_datasets[-1]
        self.val_dataset = self.train_datasets[-1]

        pad_to_multiple_of = 8 if self.args.precision in ["16-mixed", "bf16-mixed"] else None

        if self.args.task == "pretraining":
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                pad_to_multiple_of=pad_to_multiple_of,
                mlm_probability=self.args.mlm_probability,
            )
        elif self.args.task == "question-answering":
            self.data_collator = QADataCollator(
                tokenizer=self.tokenizer,
                padding=True,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                pad_to_multiple_of=pad_to_multiple_of,
            )

    def train_dataloader(self):
        common_args = dict(
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        if self.use_n_train_datasets == 1 or self.iterator_idx >= len(self.train_datasets):
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.micro_batch_size[-1],
                **common_args,
            )
            logger_text = (
                f"Switched to final dataset with a micro-batch size {self.args.micro_batch_size[-1]}. It has a length of {len(self.train_dataset)} and sequence length of {len(self.train_dataset[0]['input_ids'])}"
                if self.iterator_idx >= len(self.train_datasets)
                else f"Using training dataset with a micro-batch size {self.args.micro_batch_size[-1]}. It has a length of {len(self.train_dataset)} and sequence length of {len(self.train_dataset[0]['input_ids'])} (Note sequence length estimate only makes sense if you have used packing)"
            )
            logger.info(
                logger_text,
                rank0_only=True,
            )
            self.iterator_idx += 1
            return dataloader

        train_dataset = self.train_datasets[self.iterator_idx]
        batch_size = self.args.micro_batch_size[self.iterator_idx]
        self.iterator_idx += 1
        logger.info(
            f"Switched to dataset {self.iterator_idx} with a micro-batch size {batch_size}. It has a length of {len(train_dataset)} and sequence length of {len(train_dataset[0]['input_ids'])}",
            rank0_only=True,
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            **common_args,
        )
        return dataloader

    def val_dataloader(self):
        common_args = dict(
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        if self.use_n_val_datasets == 1:
            return DataLoader(
                self.val_dataset,
                **common_args,
                batch_size=self.args.eval_micro_batch_size[-1],
            )

        dataloaders = []
        for i in range(self.use_n_val_datasets):
            dataloaders.append(
                DataLoader(
                    self.val_datasets[i],
                    **common_args,
                    batch_size=self.args.eval_micro_batch_size[i]
                    if self.use_n_train_datasets > 1
                    else self.args.eval_micro_batch_size[-1],
                )
            )
        return dataloaders
