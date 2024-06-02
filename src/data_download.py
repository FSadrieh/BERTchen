"""
This script allows you to download and preprocess datasets for language modeling, specifically mc4 and cc100. You can customize it to your own needs.

Example command to download cc100 for German:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8

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

DEFAULT_DATA_LOCATION = "/hpi/fs00/share/fg-demelo/small-language-models/data"

DATASET_TO_TASK = {
    "c4": "pre-training",
    "cc100": "pre-training",
    "oscar2023": "pre-training",
    "germanquad": "question-answering",
    "germeval_A": "sequence-classification",
    "germeval_B": "sequence-classification",
}


@dataclass
class Args:
    out_dir: str = field(alias="-o")

    dataset: Literal["c4", "cc100", "oscar2023", "germanquad", "germeval_A", "germeval_B"] = field(default="oscar2023")
    "HF dataset"

    max_train_size: int = field(default=-1)
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

    pre_discard_factor: float = field(default=None)
    """Percentage of the dataset to discard before any processing.
    Useful for speeding up processing of huge datasets that are not fully needed."""

    max_seq_length: int = field(default=512)
    "Maximum sequence length for tokenization."

    debug: bool = field(default=False)
    "Wait for debugger to attach."

    only_download: bool = field(default=False)
    "Only download the dataset, do not process it."

    data_location: str = field(default="default")
    "We first check if we have downloaded the data already in this location, before we download it again."

    tokenizer: str = field(default="google-bert/bert-base-cased")
    "HuggingFace tokenizer identifier."

    create_n_training_datasets: int = field(default=1, alias="--cntd")
    "Create n training datasets from the same source dataset. Each training dataset will have the double the sequence length of the prior one. Only used if task is pre-training."


def load_and_process_germaneval(
    tmp_cache_dir: str, dataset_name: str, processes: int
) -> tuple[datasets.Dataset, datasets.Dataset]:
    # Downloading this dataset is different as it is not available on HF
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
        num_proc=processes,
        delimiter="\t",
        column_names=column_names,
    )
    dev_dataset = load_dataset(
        "csv",
        split="train",
        data_files="data/dev_v1.4.tsv",
        cache_dir=tmp_cache_dir,
        num_proc=processes,
        delimiter="\t",
        column_names=column_names,
    )
    # We need to remove the rows that contain no text
    train_dataset = train_dataset.filter(lambda x: x["text"])
    dev_dataset = dev_dataset.filter(lambda x: x["text"])
    if dataset_name == "germeval_A":
        train_dataset = train_dataset.map(lambda x: {"label": int(x["relevance"])})
        dev_dataset = dev_dataset.map(lambda x: {"label": int(x["relevance"])})
        train_dataset = train_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
        dev_dataset = dev_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
    else:

        def _encode_sentiment(sentiment):
            if sentiment == "positive":
                return 2
            if sentiment == "neutral":
                return 1
            return 0

        train_dataset = train_dataset.map(lambda x: {"label": _encode_sentiment(x["sentiment"])})
        dev_dataset = dev_dataset.map(lambda x: {"label": _encode_sentiment(x["sentiment"])})
        train_dataset = train_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
        dev_dataset = dev_dataset.remove_columns(["relevance", "sentiment", "aspect", "url"])
    return (train_dataset, dev_dataset)


def process_oscar(dataset: datasets.Dataset):
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
    return (dataset, None)


def load_right_dataset(
    dataset_name: str, tmp_cache_dir: str, data_location: str, split: str, processes: int
) -> tuple[datasets.Dataset, datasets.Dataset]:
    def _load_dataset(loading_args: dict, local_path: str, hf_path: str, extra_loading_args: dict = None):
        try:
            return load_dataset("json", **loading_args, data_files={"train": local_path})
        except FileNotFoundError:
            logger.info(f"Could not find dataset in {local_path}, downloading it...")
        if extra_loading_args:
            loading_args.update(extra_loading_args)
        return load_dataset(**loading_args, path=hf_path)

    def _file_location(dataset_name: str):
        return f"{DEFAULT_DATA_LOCATION}/{dataset_name}/{split}.jsonl" if data_location == "default" else data_location

    default_loading_args = {
        "split": split,
        "cache_dir": tmp_cache_dir,
        "num_proc": processes,
    }
    if dataset_name == "c4":
        extra_loading_args = {"name": "de"}
        dataset = _load_dataset(default_loading_args, _file_location("c4"), "allenai/c4", extra_loading_args)
        return (dataset, None)
    # The german split of c4 has about 25 million rows (18 GB)
    if dataset_name == "cc100":
        extra_loading_args = {"lang": "de"}
        dataset = _load_dataset(default_loading_args, _file_location("cc100"), "cc100", extra_loading_args)
        return (dataset, None)
    # The german split of oscar has about 594.7 GB
    if dataset_name == "oscar2023":
        extra_loading_args = {"token": True, "language": "de"}
        dataset = _load_dataset(
            default_loading_args, _file_location("oscar2023"), "oscar-corpus/OSCAR-2023", extra_loading_args
        )
        return process_oscar(dataset)

    if dataset_name == "germanquad":
        dataset = _load_dataset(default_loading_args, _file_location("germanquad"), "deepset/germanquad")
        return (dataset, None)

    if dataset_name in ["germeval_A", "germeval_B"]:
        return load_and_process_germaneval(tmp_cache_dir, dataset_name, processes)


def group_lines(processes: int, dataset: datasets.Dataset):
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

    map_args = dict(batched=True, batch_size=len(dataset), remove_columns=["text", "id"], num_proc=processes)
    dataset = dataset.map(_document_grouping_f, **map_args)
    dataset = dataset.rename_column("docs", "text")
    return dataset


def make_tokenize_function(tokenizer, task_type, max_seq_length=None, truncate=True):
    def sequence_classification_tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
        )
        return {**tokenized, "labels": examples["label"]}

    def question_answering_tokenize_function(examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        start_positions, end_positions, ids, answers, context = _get_labels_for_qa(
            tokenized, examples["answers"], examples["id"], examples["context"]
        )

        return {
            "id": ids,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
            "answers": answers,
            "context": context,
            "offset_mapping": tokenized["offset_mapping"],
        }

    def pre_training_tokenize_function(examples):
        text = [example.strip() + "\n" for example in examples["text"] if example != "\n" and example != ""]
        return tokenizer(
            text,
            padding=False,
            add_special_tokens=False,
        )

    if task_type == "sequence-classification":
        return sequence_classification_tokenize_function
    if task_type == "question-answering":
        return question_answering_tokenize_function
    return pre_training_tokenize_function


def _get_labels_for_qa(inputs, answers, ids, context):
    # This function is inspired from https://huggingface.co/learn/nlp-course/chapter7/7

    start_positions = []
    end_positions = []
    output_ids = []
    output_answers = []
    output_context = []
    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = inputs["overflow_to_sample_mapping"][i]
        output_ids.append(ids[sample_idx])
        output_context.append(context[sample_idx])
        answer = answers[sample_idx]
        output_answers.append({"answers": answer, "id": str(ids[sample_idx])})
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0].strip())
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    return start_positions, end_positions, output_ids, output_answers, output_context


def make_group_text_function(max_seq_length, tokenizer):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_input_ids = []
        for input_id in examples["input_ids"]:
            concatenated_input_ids.extend(input_id)
            concatenated_input_ids.append(tokenizer.sep_token_id)
        total_length = len(concatenated_input_ids)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {}
        result["input_ids"] = [concatenated_input_ids[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        result["attention_mask"] = [[1] * len(t) for t in result["input_ids"]]
        result["token_type_ids"] = [[0] * len(t) for t in result["input_ids"]]
        return result

    return group_texts


def make_different_seq_len_packing(max_seq_length):
    def change_seq_len_packing(examples):
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

    return change_seq_len_packing


def write_to_disk(train_paragraphs, dev_paragraphs, out_dir, dev_size, part_name=None):
    logger.info(f"Example train split data: {train_paragraphs.column_names}")
    logger.info(f"len: {len(train_paragraphs)}")
    logger.info("Writing data...")
    output_dir = Path(out_dir)
    os.makedirs(str(output_dir), exist_ok=True)
    PERFORMANT_BUFFER_SIZE_BYTES = 1024 * 1024 * 100  # 100 MB
    train_file_name = "train.jsonl" if part_name is None else f"train_{part_name}.jsonl"

    train_fp = io.open(str(output_dir / train_file_name), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
    with jsonlines.Writer(train_fp, compact=True) as writer:
        writer.write_all(train_paragraphs)
    train_fp.close()

    if dev_size and dev_paragraphs:
        dev_fp = io.open(str(output_dir / "dev.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(dev_fp, compact=True) as writer:
            writer.write_all(dev_paragraphs)
        dev_fp.close()

    logger.success("Done! Enjoy your data :)")
    logger.print(output_dir)


def process_pre_training_dataset(
    dataset, tokenizer, max_seq_length, task_type, processes, dev_size, max_train_size, out_dir, create_n_training_datasets
):
    processed_dataset = tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes)
    processed_dataset = group_text(processed_dataset, max_seq_length, tokenizer)
    processed_dataset, total_len = shuffle_dataset_and_log_length(processed_dataset)
    train_paragraphs, dev_paragraphs = generate_val_split(processed_dataset, total_len, dev_size)
    if create_n_training_datasets > 1:
        # We want to split the train_paragraphs into n datasets
        end = len(train_paragraphs)
        for i in range(create_n_training_datasets - 1):
            new_max_length = max_seq_length // 2 ** (create_n_training_datasets - i - 1)
            beginning = end - len(train_paragraphs) // create_n_training_datasets
            split = train_paragraphs.select(range(beginning, end))
            write_to_disk(change_packed_seq_len(split, new_max_length), None, out_dir, dev_size=0, part_name=i)
            end = beginning
        train_paragraphs = train_paragraphs.select(range(end))

    train_paragraphs = cap_training_examples(train_paragraphs, max_train_size)
    write_to_disk(train_paragraphs, dev_paragraphs, out_dir, dev_size)


def process_fine_tuning_dataset(
    train_val_datasets, tokenizer, max_seq_length, task_type, processes, dev_size, max_train_size, out_dir
):
    processed_datasets = []
    for dataset in train_val_datasets:
        if not dataset:
            processed_datasets.append(None)
            continue
        processed_datasets.append(tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes))
    processed_datasets[0], total_len = shuffle_dataset_and_log_length(processed_datasets[0])
    if processed_datasets[1] is None:
        train_paragraphs, dev_paragraphs = generate_val_split(processed_datasets[0], total_len, dev_size)
    else:
        train_paragraphs = processed_datasets[0]
        dev_paragraphs = processed_datasets[1].select(range(dev_size))
    train_paragraphs = cap_training_examples(train_paragraphs, max_train_size)
    write_to_disk(train_paragraphs, dev_paragraphs, out_dir, dev_size)


def tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes):
    return dataset.map(
        make_tokenize_function(tokenizer, max_seq_length=max_seq_length, task_type=task_type),
        batch_size=1_000,
        batched=True,
        num_proc=processes,
        remove_columns=dataset.column_names,
    )


def generate_val_split(processed_dataset, total_len, dev_size):
    train_end_idx = total_len - dev_size
    train_paragraphs = processed_dataset.select(range(train_end_idx))
    dev_paragraphs = processed_dataset.select(range(train_end_idx, train_end_idx + dev_size))
    return train_paragraphs, dev_paragraphs


def shuffle_dataset_and_log_length(processed_dataset):
    logger.info("Shuffling and splitting into sets...")
    total_len = len(processed_dataset)
    logger.info(f"Dataset len after processing: {total_len}")
    return processed_dataset.shuffle(seed=42), total_len


def cap_training_examples(train_paragraphs, max_train_size):
    if max_train_size and len(train_paragraphs) > max_train_size:
        train_paragraphs = train_paragraphs.select(range(max_train_size))
    return train_paragraphs


def group_text(processed_dataset, max_seq_length, tokenizer):
    return processed_dataset.map(
        make_group_text_function(max_seq_length, tokenizer),
        batch_size=1_000,
        batched=True,
        remove_columns=processed_dataset.column_names,
    )


def change_packed_seq_len(processed_dataset, max_seq_length):
    return processed_dataset.map(
        make_different_seq_len_packing(max_seq_length),
        batch_size=1_000,
        batched=True,
        remove_columns=processed_dataset.column_names,
    )


@graceful_exceptions()
def main(args: Args):
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
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
    train_val_datasets = load_right_dataset(args.dataset, tmp_cache_dir, args.data_location, args.split, args.processes)

    ### Pre discard some data ###
    if args.pre_discard_factor:
        train_val_datasets = [
            train_val_datasets[0]
            .shuffle(seed=42)
            .select(range(int((1 - args.pre_discard_factor) * len(train_val_datasets[0])))),
            train_val_datasets[1]
            .shuffle(seed=42)
            .select(range(int((1 - args.pre_discard_factor) * len(train_val_datasets[1]))))
            if train_val_datasets[1]
            else None,
        ]

    #### Only download ####
    if args.only_download:
        write_to_disk(train_val_datasets[0], train_val_datasets[1], args.out_dir, args.dev_size)
        return

    ##### For CC100: Group individual lines into documents #####
    train_val_datasets = (
        (group_lines(args.processes, train_val_datasets[0]), None) if args.dataset == "cc100" else train_val_datasets
    )

    logger.info("Starting mapping & chunking")
    process_args = {
        "max_seq_length": args.max_seq_length,
        "tokenizer": tokenizer,
        "task_type": DATASET_TO_TASK[args.dataset],
        "processes": args.processes,
        "dev_size": args.dev_size,
        "max_train_size": args.max_train_size,
        "out_dir": args.out_dir,
    }
    if DATASET_TO_TASK[args.dataset] == "pre-training":
        process_pre_training_dataset(
            train_val_datasets[0], **process_args, create_n_training_datasets=args.create_n_training_datasets
        )
    else:
        process_fine_tuning_dataset(train_val_datasets, **process_args)

    if args.conserve_disk_space:
        logger.info("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise


if __name__ == "__main__":
    args = parse(Args)
    if args.debug:
        wait_for_debugger()
    main(args)
