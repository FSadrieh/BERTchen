"""
This script allows you to download and preprocess datasets for language modeling, specifically mc4 and cc100. You can customize it to your own needs.

Example command to download cc100 for German:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8


Example command to download cc100 for German using streaming mode for HF datasets (faster, requires less RAM) and cleaning up caches:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8 --stream --stream_shuffle_buffer_size 10_000 --conserve_disk_space

Inspiration from lit-gpt and gpt-neox.
"""

import errno
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal
from itertools import chain

import datasets
import jsonlines
from datasets import load_dataset
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import field, parse
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils import wait_for_debugger


PRE_TRAINING_DATASETS = ["c4", "cc100", "oscar2023"]


@dataclass
class Args:
    out_dir: str = field(alias="-o")

    dataset: Literal["c4", "cc100", "oscar2023", "germanquad", "germeval_A", "germeval_B"] = field(default="oscar2023")
    "HF dataset"

    max_train_size: int = field(default=50_000)
    "Maximum number of train documents to write to disk. Use to limit very large datasets that you will not exhaust during training anyway. Use -1 to disable."

    dev_size: int = field(default=2_500)
    "If 0, do not construct dev set."

    test_size: int = field(default=0)
    "If 0, do not contruct test set."

    processes: int = field(default=4)
    "Number of processes for parallel tokenization."

    split: str = field(default="train")
    "Select percentage of dataset like so: --split=train[:50%]"

    conserve_disk_space: bool = field(default=False, alias="--disk_space")
    "Disable all HF caching and cleanup download caches to conserve disk space."

    stream_shuffle_buffer_size: int = field(default=100_000)
    """Buffer size for shuffling datasets before splitting in streaming mode.
    The entire buffer will be downloaded before shuffling.
    You also need to have enough RAM if you set this to a large value If -1, set to max_train_size."""

    pre_discard_factor: float = field(default=None)
    """Percentage of the dataset to discard before any processing.
    Useful for speeding up processing of huge datasets that are not fully needed.
    Not needed if you use --stream."""

    max_seq_length: int = field(default=512)
    "Maximum sequence length for tokenization."

    debug: bool = field(default=False)
    "Wait for debugger to attach."


def load_right_dataset(
    dataset_name: str, tmp_cache_dir: str, stream: bool, args: Args
) -> tuple[datasets.Dataset, datasets.Dataset]:
    if dataset_name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            "de",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=stream,
            num_proc=None if stream else args.processes,
        )
        train_val_datasets = (dataset, None)
    # The german split of c4 has about 25 million rows (18 GB)
    elif dataset_name == "cc100":
        if stream:
            logger.warning("Streaming mode for cc100 might lose some sample documents.")
            # Streaming mode is not trivial for cc100, since we need to group samples into documents.
            # To lose no samples, we need to set batch_size=len(dataset) but this is not possible for IteratableDataset.
            # We can accept loosing some samples by setting batch_size to a large number.
        dataset = load_dataset(
            "cc100",
            lang="de",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=stream,
            num_proc=None if stream else args.processes,
        )
        train_val_datasets = (dataset, None)
    # The german split of oscar has about 594.7 GB
    elif dataset_name == "oscar2023":
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            language="de",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=stream,
            token=True,
            num_proc=None if stream else args.processes,
        )

        # if dataset_name == "oscar2023":
        # For oscar2023, we need to rename the columns to match the other datasets
        # dataset = dataset.rename_column("content", "text")

        # Filter out all samples with content warning in OSCAR
        dataset = dataset.filter(
            lambda x: x["meta"]["quality_warnings"] is None
            or (
                "noisy" not in x["meta"]["quality_warnings"]
                and "header" not in x["meta"]["quality_warnings"]
                and "footer" not in x["meta"]["quality_warnings"]
                and "short_sentences" not in x["meta"]["quality_warnings"]
                and "tiny" not in x["meta"]["quality_warnings"]
                and "adult" not in x["meta"]["quality_warnings"]
            ),
        )
        train_val_datasets = (dataset, None)

    elif dataset_name == "germanquad":
        dataset = load_dataset(
            "deepset/germanquad",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=stream,
            num_proc=None if stream else args.processes,
        )
        dataset = dataset.map(lambda x: {"text": "Kontext: " + x["context"] + ";\nFrage: " + x["question"]})
        dataset = dataset.map(lambda x: {"label": x["answers"]["text"][0]})
        dataset = dataset.remove_columns(["id", "context", "question", "answers"])
        train_val_datasets = (dataset, None)

    elif dataset_name in ["germeval_A", "germeval_B"]:
        if not (os.path.exists("data/dev_v1.4.tsv") and os.path.exists("data/train_v1.4.tsv")):
            raise FileNotFoundError(
                "Please download the GermEval dataset from https://sites.google.com/view/germeval2017-absa/data and place it in the data/ folder."
            )
        column_names = ["url", "text", "relevance", "sentiment", "aspect"]
        train_dataset = load_dataset(
            "csv",
            split="train",
            data_files="data/train_v1.4.tsv",
            cache_dir=tmp_cache_dir,
            num_proc=args.processes,
            delimiter="\t",
            column_names=column_names,
        )
        dev_dataset = load_dataset(
            "csv",
            split="train",
            data_files="data/dev_v1.4.tsv",
            cache_dir=tmp_cache_dir,
            num_proc=args.processes,
            delimiter="\t",
            column_names=column_names,
        )
        # We need to remove the rows that contain no text
        train_dataset = train_dataset.filter(lambda x: x["text"])
        dev_dataset = dev_dataset.filter(lambda x: x["text"])
        if dataset_name == "germeval_A":
            train_dataset = train_dataset.map(lambda x: {"label": "Ja" if x["relevance"] else "Nein"})
            dev_dataset = dev_dataset.map(lambda x: {"label": "Ja" if x["relevance"] else "Nein"})
            train_dataset = train_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
            dev_dataset = dev_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
        else:
            train_dataset = train_dataset.map(lambda x: {"label": x["sentiment"]})
            dev_dataset = dev_dataset.map(lambda x: {"label": x["sentiment"]})
            train_dataset = train_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
            dev_dataset = dev_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])

        train_val_datasets = (train_dataset, dev_dataset)
    return train_val_datasets


def group_lines(args, dataset: datasets.Dataset):
    def _document_grouping_f(examples: Dict[str, list[str]]):
        documents = []
        current_doc = ""
        for example in examples["text"]:
            if example == "\n":
                documents.append(current_doc)
                current_doc = ""
            else:
                current_doc += example
        return {"docs": documents}

    batch_size = 16_000 if args.stream else len(dataset)
    map_args = dict(batched=True, batch_size=batch_size, remove_columns=["text", "id"])
    if not args.stream:
        map_args["num_proc"] = args.processes  # stream does not support multiprocessing
    dataset = dataset.map(_document_grouping_f, **map_args)
    dataset = dataset.rename_column("docs", "text")
    return dataset


def make_tokenize_function(tokenizer, max_seq_length=None, truncate=True, is_pre_training: bool = False):
    def fine_tuning_tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
        )
        tokenized_labels = tokenizer(
            examples["label"],
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        return {**tokenized, "labels": tokenized_labels["input_ids"]}

    def pre_training_tokenize_function(examples):
        text = [example.strip() + "\n" for example in examples["text"] if example != "\n"]
        return tokenizer(
            text,
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
        )

    return pre_training_tokenize_function if is_pre_training else fine_tuning_tokenize_function


def make_group_text_function(max_seq_length):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    return group_texts


@graceful_exceptions()
def main(args: Args):
    # As long as we do not have trained our own tokenizer we are using the huggingface one
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("google-bert/bert-base-cased", use_fast=True)
    logger.info(args)
    if args.max_train_size == -1:
        args.max_train_size = None
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        logger.info("Disabling caching to conserve disk space.")
        datasets.fingerprint.disable_caching()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("Downloading dataset. This can take some time, so sit back and relax...")

    tmp_cache_dir = None
    if args.conserve_disk_space:
        tmp_cache_dir = os.path.join(args.out_dir, args.language, "tmp_download_cache")
        os.makedirs(tmp_cache_dir, exist_ok=True)

    ##### Load dataset #####
    stream = args.dataset in PRE_TRAINING_DATASETS
    train_val_datasets = load_right_dataset(args.dataset, tmp_cache_dir, stream, args)

    ##### For CC100: Group individual lines into documents #####
    train_val_datasets = (group_lines(args, train_val_datasets[0]), None) if args.dataset == "cc100" else train_val_datasets

    ##### Process dataset #####
    logger.info("Starting mapping & chunking")
    processed_datasets = []
    for dataset in train_val_datasets:
        if not dataset:
            processed_datasets.append(None)
            continue
        processed_dataset = dataset.map(
            make_tokenize_function(tokenizer, max_seq_length=args.max_seq_length),
            batch_size=1_000,
            batched=True,
            num_proc=None if stream else args.processes,
            remove_columns=dataset.column_names,
        )
        logger.info("Tokenization finished!")
        if args.dataset in PRE_TRAINING_DATASETS:
            processed_dataset = processed_dataset.map(
                make_group_text_function(args.max_seq_length),
                batch_size=1_000,
                batched=True,
                remove_columns=processed_dataset.column_names,
            )
        processed_datasets.append(processed_dataset)
    logger.success("Processing finished!")

    ##### Split into train/dev/test #####
    logger.info("Shuffling and splitting into sets...")
    if stream:
        # Careful here, this does not truly shuffle ALL the data by default, only samples within a buffer
        # You might have to adjust the buffer_size here depending on memory limits of your machine
        # Then take care of true shuffling in the Dataloader
        # Used for pretraining datasets, thus no need for handling the dev set
        processed_dataset = processed_datasets[0]
        args.stream_shuffle_buffer_size = None if args.stream_shuffle_buffer_size == -1 else args.stream_shuffle_buffer_size
        logger.debug(f"Shuffling with buffer size {args.stream_shuffle_buffer_size}")
        dataset = processed_dataset.shuffle(seed=42, buffer_size=args.stream_shuffle_buffer_size)

        if args.dev_size:
            logger.debug(f"Taking {args.dev_size} dev samples")
            dev_paragraphs = processed_dataset.take(args.dev_size)
            processed_dataset = processed_dataset.skip(args.dev_size)

        if args.test_size:
            logger.debug(f"Taking {args.test_size} test samples")
            test_paragraphs = processed_dataset.take(args.test_size)
            processed_dataset = processed_dataset.skip(args.test_size)

        logger.debug(f"Taking {args.max_train_size} train samples")
        train_paragraphs = processed_dataset.take(args.max_train_size)

        logger.info(f"Example train split data: {list(train_paragraphs.take(4))}")
    else:
        total_len = len(processed_datasets[0])
        logger.info(f"Dataset len after processing: {total_len}")

        processed_datasets[0] = processed_datasets[0].shuffle(seed=42)

        if not processed_datasets[1]:
            processed_dataset = processed_datasets[0]
            dev_test_size = args.dev_size + (args.test_size or 0)
            train_end_idx = total_len - dev_test_size
            train_paragraphs = processed_dataset.select(range(train_end_idx))
            dev_paragraphs = processed_dataset.select(range(train_end_idx, train_end_idx + args.dev_size))
        else:
            train_paragraphs = processed_datasets[0]
            dev_paragraphs = processed_datasets[1].select(range(args.dev_size))

        if args.max_train_size and len(train_paragraphs) > args.max_train_size:
            train_paragraphs = train_paragraphs.select(range(args.max_train_size))

        logger.info(f"Example train split data: {train_paragraphs[:4]}")
        logger.info(f"len: {len(train_paragraphs)}")

    if args.conserve_disk_space:
        logger.info("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise

    ##### Write to disk #####
    logger.info("Writing data...")
    output_dir = Path(args.out_dir)
    os.makedirs(str(output_dir), exist_ok=True)
    PERFORMANT_BUFFER_SIZE_BYTES = 1024 * 1024 * 100  # 100 MB

    train_fp = io.open(str(output_dir / "train.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
    with jsonlines.Writer(train_fp, compact=True) as writer:
        writer.write_all(train_paragraphs)
    train_fp.close()

    if args.dev_size:
        dev_fp = io.open(str(output_dir / "dev.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(dev_fp, compact=True) as writer:
            writer.write_all(dev_paragraphs)
        dev_fp.close()

    if args.test_size:
        test_fp = io.open(str(output_dir / "test.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(test_fp, compact=True) as writer:
            writer.write_all(test_paragraphs)
        test_fp.close()

    logger.success("Done! Enjoy your data :)")
    logger.print(output_dir / "train.jsonl")


if __name__ == "__main__":
    args = parse(Args)
    if args.debug:
        wait_for_debugger()
    main(args)
