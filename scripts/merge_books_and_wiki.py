"""
    This script merges the wiki and books datasets into a single dataset. We assume both have been downloaded (For wiki see src/data_download.py and for books please download from https://figshare.com/articles/dataset/Corpus_of_German-Language_Fiction_txt_/4524680).
    The book dataset is one directory of many txt files, while the wiki dataset is a single jsonl file. We merge them into a single dataset.
"""

from dataclasses import dataclass
from pathlib import Path
import os
from fnmatch import fnmatch
from tqdm import tqdm

import jsonlines
from datasets import load_dataset
from print_on_steroids import logger
from simple_parsing import field, parse

@dataclass
class Args:
    out_dir: str = field(alias="-o")

    books_dir: str = field(alias="-b")

    wiki_path: str = field(alias="-w", default=None)

    only_books: bool = field(alias="-ob", default=False)


def main():
    args = parse(Args)
    if not args.only_books:
        wiki_dataset = load_dataset("json", data_files=args.wiki_path)
        logger.info(f"Loaded wiki dataset with {len(wiki_dataset['train'])} examples")

    books_dataset = []
    for path, subdirs, files in os.walk(args.books_dir):
        for name in tqdm(files):
            if fnmatch(name, "*txt"):
                with open(os.path.join(path, name), "r") as f:
                    book = f.read()
                    book = book.replace("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", "")
                    books_dataset.append(book)

    logger.info(f"Loaded book dataset with {len(books_dataset)} examples")


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_dir / "train.jsonl", "w") as writer:
        for book in books_dataset:
            writer.write({"text": book})

        if not args.only_books:
            for wiki in wiki_dataset["train"]:
                writer.write({"text": wiki["text"]})

    logger.info(f"Saved merged dataset to {out_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
