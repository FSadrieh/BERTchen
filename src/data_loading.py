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
    ):
        super().__init__()
        self.args = training_args
        self.data_dir = training_args.data_dir
        train_file, val_file = (
            self.data_dir / self.args.train_file,
            self.data_dir / self.args.val_file,
        )

        logger.debug(f"Train file path: {train_file} val file path: {val_file}")

        self.train_file = str(train_file)
        self.val_file = str(val_file)
        self.local_rank = get_rank()

        self.tokenizer = tokenizer
        self.iterator_idx = 0
        self.use_n_training_datasets = self.args.use_n_training_datasets

    def prepare_data(self) -> None:
        if not (os.path.exists(self.train_file) and os.path.exists(self.val_file)):
            logger.info(f"Could not find processed dataset: {self.train_file}, please create it via data download")

    def setup(self, stage):
        logger.info(f"Loading cached processed dataset from {self.data_dir}...", rank0_only=False)
        train_val_datasets = datasets.load_dataset(
            "json",
            data_files={"train": self.train_file, "val": self.val_file},
            name=str(self.data_dir).replace("/", "_"),
            num_proc=self.args.preprocessing_workers,
        )
        self.train_dataset = train_val_datasets["train"]
        self.val_dataset = train_val_datasets["val"]

        if self.use_n_training_datasets > 1:
            self.train_datasets = []
            for i in range(self.use_n_training_datasets - 1):
                train_file = str(self.data_dir / f"train_{i}.jsonl")
                train_dataset = datasets.load_dataset(
                    "json",
                    data_files={"train": train_file},
                    name=str(self.data_dir).replace("/", "_"),
                    num_proc=self.args.preprocessing_workers,
                )
                self.train_datasets.append(train_dataset["train"])

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

        if self.use_n_training_datasets == 1 or self.iterator_idx >= len(self.train_datasets):
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.micro_batch_size[-1],
                **common_args,
            )
            logger.info(
                f"Switched to final dataset with a micro-batch size {self.args.micro_batch_size[-1]}. It has a length of {len(self.train_dataset)} and sequence length of {len(self.train_dataset[0]['input_ids'])}",
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
            batch_size=self.args.eval_micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.val_dataset, collate_fn=self.data_collator, **common_args, shuffle=False)
