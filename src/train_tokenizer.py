from dataclasses import dataclass
from typing import Literal

from simple_parsing import field, parse
from transformers import AutoTokenizer

from utils import wait_for_debugger
from data_download import load_right_dataset


@dataclass
class Args:
    out_dir: str = field(alias="-o")

    dataset: Literal["c4", "cc100", "oscar2023", "germanquad", "germeval_A", "germeval_B"] = field(default="oscar2023")
    "HF dataset"

    debug: bool = field(default=False)
    "Wait for debugger to attach."

    data_location: str = field(default="default")
    "We first check if we have downloaded the data already in this location, before we download it again."

    processes: int = field(default=4)

    max_vocab_size: int = field(default=30_522)

    tokenizer_name: str = field(default="bert-base-cased")


def get_training_corpus(dataset):
    # Taken from https://huggingface.co/learn/nlp-course/chapter6/2
    return (dataset[i : i + 1000]["text"] for i in range(0, len(dataset), 1000))


def main(args: Args):
    untrained_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    dataset = load_right_dataset(args.dataset, None, args.data_location, "train", args.processes)[0]
    training_corpus = get_training_corpus(dataset)
    tokenizer = untrained_tokenizer.train_new_from_iterator(training_corpus, vocab_size=args.max_vocab_size)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    args = parse(Args)
    if args.debug:
        wait_for_debugger()
    main(args)
