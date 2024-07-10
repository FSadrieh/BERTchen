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
import yaml


import datasets
import jsonlines
from datasets import load_dataset
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import field, parse
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils import wait_for_debugger

DEFAULT_DATA_LOCATION = "/hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data"

DATASET_TO_TASK = {
    "c4": "pre-training",
    "cc100": "pre-training",
    "CulturaX": "pre-training",
    "germanquad": "question-answering",
    "germeval_A": "sequence-classification",
    "germeval_B": "sequence-classification",
    "germeval_24": "sequence-classification",
    "wikipedia": "pre-training",
    "oscar2023": "pre-training",
}


@dataclass
class Args:
    out_dir: str = field(alias="-o")

    dataset: Literal[
        "c4", "cc100", "CulturaX", "germanquad", "germeval_A", "germeval_B", "germeval_24", "wikipedia", "oscar2023"
    ] = field(default="CulturaX")
    "HF dataset"

    max_train_size: int = field(default=-1)
    "Maximum number of train documents to write to disk. Use to limit very large datasets that you will not exhaust during training anyway. Use -1 to disable."

    dev_size: int = field(default=2_500)
    "If 0, do not construct dev set. -1 means use all data if a dev_set is available."

    test_size: int = field(default=0)
    "If 0, do not construct test set."

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

    tokenizer: str = field(default="tokenizer")
    "HuggingFace tokenizer identifier."

    train_sequence_lengths: str = field(default=None, alias="--tsl")
    "Specify the sequence lengths of the training datasets seperated by comma. If None only create one training set with maximum seq len. Only used if task is pre-training."

    val_sequence_lengths: str = field(default=None, alias="--vsl")
    "Specify the sequence lengths of the validation datasets seperated by comma. If None only create one validation set with maximum seq len Only used if task is pre-training."

    dataset_size_splits: str = field(default=None)
    "If set it will determine, the sizes of the different train datasets. Format: 0.5,0.25,0.25 (For len(train_sequence_lengths) = 3)"

    strategy: Literal["naive", "sorted", "not_packed"] = field(default="naive")
    "Strategy to split the dataset into different training datasets. Only used if task is pre-training and we have more than one datasets."

    get_stats: bool = field(default=False)
    "Will load the dataset and print some statistics about it."

    truncate: bool = field(default=True)
    "Truncate sequences to max_seq_length. Only for fine-tuning."


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


def load_and_process_germeval_24(tmp_cache_dir: str, processes: int) -> datasets.Dataset:
    SEXISM_LEVELS = {
        "0-Kein": 0,
        "1-Gering": 1,
        "2-Vorhanden": 2,
        "3-Stark": 3,
        "4-Extrem": 4,
    }

    def _encode_sexism_level(annotations):
        """Encodes the sexism level from the annotations as a majority vote. If there is no majority, we return the lowest level."""
        votes = [SEXISM_LEVELS[annotation["label"]] for annotation in annotations]
        return max(set(votes), key=votes.count)

    # Downloading this dataset is different as it is not available on HF
    if not (os.path.exists("data/dev_v1.4.tsv") and os.path.exists("data/train_v1.4.tsv")):
        raise FileNotFoundError(
            "Please download the competition phase training data of the GermEval dataset from https://ofai.github.io/GermEval2024-GerMS/download.html and place it in the data/ folder."
        )
    dataset = load_dataset(
        "json",
        data_files={"train": "data/germeval_24.jsonl"},
        cache_dir=tmp_cache_dir,
        num_proc=processes,
    )["train"]

    dataset = dataset.map(lambda x: {"label": _encode_sexism_level(x["annotations"])})

    return dataset


def load_right_dataset(
    dataset_name: str, tmp_cache_dir: str, data_location: str, split: str, processes: int
) -> tuple[datasets.Dataset, datasets.Dataset]:
    def _load_dataset(loading_args: dict, local_path: str, hf_path: str, extra_loading_args: dict = None, token=False):
        try:
            return load_dataset("json", **loading_args, data_files={"train": local_path})
        except FileNotFoundError:
            logger.info(f"Could not find dataset in {local_path}, downloading it...")
        if extra_loading_args:
            loading_args.update(extra_loading_args)
        return load_dataset(**loading_args, path=hf_path, token=token)

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
        dataset = dataset.select_columns(["text"])
        return (dataset, None)
    if dataset_name == "CulturaX":
        extra_loading_args = {"name": "de"}
        dataset = _load_dataset(
            default_loading_args, _file_location("CulturaX"), "uonlp/CulturaX", extra_loading_args, token=True
        )
        dataset = dataset.select_columns(["text"])
        return (dataset, None)
    if dataset_name == "cc100":
        extra_loading_args = {"lang": "de"}
        dataset = _load_dataset(default_loading_args, _file_location("cc100"), "cc100", extra_loading_args)
        return (dataset, None)

    if dataset_name == "wikipedia": # If you have downloaded the books dataset, you can use wikipedia as a dataset name
        extra_loading_args = {"name": "20220301.de"}
        dataset = _load_dataset(default_loading_args, _file_location("wikipedia"), "wikipedia", extra_loading_args)
        return (dataset, None)

    if dataset_name == "oscar2023":
        extra_loading_args = {"language": "de"}
        dataset = _load_dataset(
            default_loading_args, _file_location("oscar2023"), "oscar-corpus/OSCAR-2301", extra_loading_args, token=True
        )

        if "meta" in dataset.column_names:
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
            dataset = dataset.select_columns(["text"])
        return (dataset, None)

    if dataset_name == "germanquad":
        dataset = _load_dataset(default_loading_args, _file_location("germanquad"), "deepset/germanquad")
        return (dataset, None)

    if dataset_name in ["germeval_A", "germeval_B"]:
        return load_and_process_germaneval(tmp_cache_dir, dataset_name, processes)

    if dataset_name == "germeval_24":
        dataset = load_and_process_germeval_24(tmp_cache_dir, processes)
        return (dataset, None)


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
    dataset = dataset.map(_document_grouping_f, **map_args, desc="Grouping lines for cc100")
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
        if truncate:
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
        else:
            tokenized = tokenizer(
                questions,
                examples["context"],
            )
            start_positions, end_positions, ids, answers, context = _get_untruncated_labels_for_qa(
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
            "offset_mapping": tokenized["offset_mapping"], #TODO: OFFSET MAPPING DOES NOT NEED TO BE SPECIFIED FIX AND TEST RIGHT ANSWERS
        }

    def pre_training_tokenize_function(examples):
        text = [example.strip() + "\n" for example in examples["text"] if example != "\n" and example != ""]
        return tokenizer(
            text,
            padding=False,
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
            if idx == len(sequence_ids):
                break
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

def _get_untruncated_labels_for_qa(inputs, answers, ids, context):
    start_positions = []
    end_positions = []
    output_answers = []
    for i in range(len(inputs["input_ids"])):
        answer = answers[i]
        output_answers.append({"answers": answer, "id": str(ids[i])})
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        start_positions.append(answer["answer_start"][0] + context_start)
        end_positions.append(answer["answer_start"][0] + len(answer["text"][0].strip()) + context_start)
    return start_positions, end_positions, ids, output_answers, context


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


def write_to_disk(train_paragraphs, dev_paragraphs, out_dir, part_name=None):
    assert train_paragraphs or dev_paragraphs, "At least one of train or dev paragraphs must be provided."
    logger.info(f"Example train split data: {train_paragraphs.column_names if train_paragraphs else dev_paragraphs}")
    logger.info(f"len: {len(train_paragraphs) if train_paragraphs else len(dev_paragraphs)}")
    logger.info("Writing data...")
    output_dir = Path(out_dir)
    os.makedirs(str(output_dir), exist_ok=True)
    PERFORMANT_BUFFER_SIZE_BYTES = 1024 * 1024 * 100  # 100 MB
    train_file_name = "train.jsonl" if part_name is None else f"train_{part_name}.jsonl"

    if train_paragraphs is not None:
        train_fp = io.open(str(output_dir / train_file_name), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(train_fp, compact=True) as writer:
            writer.write_all(train_paragraphs)
        train_fp.close()

    if dev_paragraphs is not None:
        dev_file_name = "dev.jsonl" if part_name is None else f"dev_{part_name}.jsonl"
        dev_fp = io.open(str(output_dir / dev_file_name), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(dev_fp, compact=True) as writer:
            writer.write_all(dev_paragraphs)
        dev_fp.close()

    logger.success("Done! Enjoy your data :)")
    logger.print(output_dir)


def get_stats(dataset):
    logger.info(f"Number of examples: {len(dataset)}")
    # Get min, max, mean, median, std of the sequence length
    dataset = dataset.map(
        lambda x: {"length": len(x["input_ids"])}, desc="Calculating lengths", remove_columns=dataset.column_names
    )
    dataset = dataset.sort("length")
    logger.info(f"Min sequence length: {dataset['length'][0]}")
    logger.info(f"Max sequence length: {dataset['length'][-1]}")
    length = len(dataset)
    sum_len = sum(dataset["length"])
    logger.info(f"Mean sequence length: {sum_len / length}")
    logger.info(f"Median sequence length: {dataset[length // 2]}")

def process_pre_training_dataset(
    dataset,
    tokenizer,
    max_seq_length,
    task_type,
    processes,
    dev_size,
    max_train_size,
    out_dir,
    training_seq_lens,
    validation_seq_lens,
    dataset_size_splits,
    strategy,
    stats,
):
    content = {}
    processed_dataset = tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes)
    if stats:
        get_stats(processed_dataset)
        exit(0)

    if strategy == "sorted":
        processed_dataset = sort_dataset_by_length(processed_dataset)
    if strategy != "not_packed":
        processed_dataset = group_text(processed_dataset, max_seq_length, tokenizer)
        if strategy == "sorted":
            total_len = len(processed_dataset)
        else:
            processed_dataset, total_len = shuffle_dataset_and_log_length(processed_dataset)
        train_paragraphs, dev_paragraphs = generate_val_split(processed_dataset, total_len, dev_size)
        if len(training_seq_lens) > 1:
            # We want to split the train_paragraphs into n datasets
            dataset_splits = split_dataset(train_paragraphs, dataset_size_splits, training_seq_lens, max_seq_length, strategy)
            for i, split in enumerate(dataset_splits):
                if i == len(training_seq_lens) - 1:
                    split = cap_training_examples(split, max_train_size)
                write_to_disk(split, None, out_dir, part_name=i)
                content[f"train_{i}.jsonl"] = {"seq_len": training_seq_lens[i], "size": len(split)}

        else:
            if training_seq_lens[0] != max_seq_length:
                train_paragraphs = change_packed_seq_len(train_paragraphs, training_seq_lens[0])
            train_paragraphs = cap_training_examples(train_paragraphs, max_train_size)
            write_to_disk(train_paragraphs, None, out_dir)
            content["train.jsonl"] = {"seq_len": training_seq_lens[0], "size": len(train_paragraphs)}
    else:
        content, dev_paragraphs = select_sequence_lengths(processed_dataset, training_seq_lens, out_dir, content, dev_size)

    if len(validation_seq_lens) > 1:
        # We want to put the same dev_paragraphs into n datasets with differing sequence length
        for i, seq_len in enumerate(validation_seq_lens):
            if seq_len != max_seq_length:
                dataset = change_packed_seq_len(dev_paragraphs, seq_len)
            else:
                dataset = dev_paragraphs
            write_to_disk(None, dataset, out_dir, part_name=i)
            content[f"dev_{i}.jsonl"] = {"seq_len": seq_len, "size": len(dataset)}

    else:
        if validation_seq_lens[0] != max_seq_length:
            dev_paragraphs = change_packed_seq_len(dev_paragraphs, validation_seq_lens[0])
        write_to_disk(None, dev_paragraphs, out_dir)
        content["dev.jsonl"] = {"seq_len": validation_seq_lens[0], "size": len(dev_paragraphs)}

    # Write yml file indicating which dataset has which sequence length and size
    if content:
        with open(os.path.join(out_dir, "dataset_info.yml"), "w") as f:
            yaml.dump(content, f)

def select_sequence_lengths(processed_dataset, training_seq_lens, out_dir, content, dev_size):
    """
    This function selects the sequence lengths for the different training datasets. We select the right sequence length to avoid packing and context fragmentation.
    """
    dev_paragraphs = processed_dataset.filter(
        lambda x: len(x["input_ids"]) not in training_seq_lens,
        desc="Selecting sequence lengths for dev set.",
    )
    train_paragraphs = processed_dataset.filter(
        lambda x: len(x["input_ids"]) in training_seq_lens,
        desc="Selecting sequence lengths for train set.",
    )
    
    logger.info(f"Number of examples in Dataset: {len(processed_dataset)} of those {len(dev_paragraphs)} have the wrong sequence length. We will use them for the dev set. {len(train_paragraphs)} will be used for training.")
    __, dev_paragraphs = generate_val_split(dev_paragraphs, len(dev_paragraphs), dev_size)
    for i, training_seq_len in enumerate(training_seq_lens):
        split = train_paragraphs.filter(lambda x: len(x["input_ids"][0]) == training_seq_len)
        if len(split) > 0:
            write_to_disk(split, None, out_dir, part_name=i)
            content[f"train_{i}.jsonl"] = {"seq_len": training_seq_lens[i], "size": len(split)}

    return content, dev_paragraphs

def split_dataset(train_paragraphs, dataset_size_splits, training_seq_lens, max_seq_length, strategy):
    """
    This function selects the splits for the different training datasets.
    In the future it should support different strategies.
    """
    splits = []
    end = len(train_paragraphs)
    for i in range(len(training_seq_lens) - 1):
        beginning = int(end - len(train_paragraphs) * dataset_size_splits[i])
        split = train_paragraphs.select(range(beginning, end))
        if training_seq_lens[i] != max_seq_length:
            split = change_packed_seq_len(split, training_seq_lens[i])
        splits.append(split)
        end = beginning
    split = train_paragraphs.select(range(end))
    if training_seq_lens[-1] != max_seq_length:
        split = change_packed_seq_len(split, training_seq_lens[-1])
    splits.append(split)

    return splits


def sort_dataset_by_length(dataset):
    def _length(x):
        x["length"] = [len(i) for i in x["input_ids"]]
        return x

    dataset_with_lengths = dataset.map(_length, batched=True, desc="Calculating lengths")
    dataset_with_lengths = dataset_with_lengths.sort("length", reverse=True)
    return dataset_with_lengths


def process_fine_tuning_dataset(
    train_val_datasets, tokenizer, max_seq_length, task_type, processes, dev_size, max_train_size, out_dir, stats, truncate
):
    processed_datasets = []
    for dataset in train_val_datasets:
        if not dataset:
            processed_datasets.append(None)
            continue
        processed_datasets.append(tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes, truncate))
    processed_datasets[0], total_len = shuffle_dataset_and_log_length(processed_datasets[0])
    if processed_datasets[1] is None:
        train_paragraphs, dev_paragraphs = generate_val_split(processed_datasets[0], total_len, dev_size)
    else:
        train_paragraphs = processed_datasets[0]
        dev_paragraphs = processed_datasets[1].select(range(dev_size)) if dev_size != -1 else processed_datasets[1]
    if stats:
        logger.info("Calculating statistics for train set")
        get_stats(train_paragraphs)
        logger.info("Calculating statistics for dev set")
        get_stats(dev_paragraphs)
        exit(0)
    train_paragraphs = cap_training_examples(train_paragraphs, max_train_size)
    write_to_disk(train_paragraphs, dev_paragraphs, out_dir)


def tokenize_dataset(dataset, tokenizer, max_seq_length, task_type, processes, truncate=False):
    return dataset.map(
        make_tokenize_function(tokenizer, max_seq_length=max_seq_length, task_type=task_type, truncate=truncate),
        batch_size=16_000,
        batched=True,
        num_proc=processes,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
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
        batch_size=16_000,
        batched=True,
        remove_columns=processed_dataset.column_names,
        desc="Grouping texts",
    )


def change_packed_seq_len(processed_dataset, max_seq_length):
    return processed_dataset.map(
        make_different_seq_len_packing(max_seq_length),
        batch_size=16_000,
        batched=True,
        remove_columns=processed_dataset.column_names,
        desc=f"Changing packed sequence length to {max_seq_length}",
    )


@graceful_exceptions()
def main(args: Args):
    if args.get_stats:
        args.out_dir = "stats"
        args.only_download = False
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    logger.info(args)
    if args.max_train_size == -1:
        args.max_train_size = None
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        logger.info("Disabling caching to conserve disk space.")
        datasets.fingerprint.disable_caching()

    training_seq_lens = (
        list(map(int, str(args.train_sequence_lengths).split(","))) if args.train_sequence_lengths else [args.max_seq_length]
    )
    dataset_size_splits = (
        list(map(float, str(args.dataset_size_splits).split(",")))
        if args.dataset_size_splits
        else [1 / len(training_seq_lens)] * len(training_seq_lens)
    )
    validation_seq_lens = (
        list(map(int, str(args.val_sequence_lengths).split(","))) if args.val_sequence_lengths else [args.max_seq_length]
    )
    assert sum(dataset_size_splits) == 1, "The sum of the dataset size splits must be 1."
    assert len(dataset_size_splits) == len(
        training_seq_lens
    ), "The number of dataset size splits must be equal to the number of training datasets."

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
        write_to_disk(train_val_datasets[0], train_val_datasets[1], args.out_dir)
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
        "stats": args.get_stats,
    }
    if DATASET_TO_TASK[args.dataset] == "pre-training":
        process_pre_training_dataset(
            train_val_datasets[0],
            **process_args,
            training_seq_lens=training_seq_lens,
            validation_seq_lens=validation_seq_lens,
            dataset_size_splits=dataset_size_splits,
            strategy=args.strategy,
        )
    else:
        process_fine_tuning_dataset(train_val_datasets, **process_args, truncate=args.truncate)

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
    args = parse(Args, add_config_path_arg=True)
    if args.debug:
        wait_for_debugger()
    main(args)
    # We want to save the hyperparameters of this data download
    content = vars(args)
    with open(os.path.join(args.out_dir, "dataset_hyperparams.yml"), "w") as f:
        yaml.dump(content, f)
