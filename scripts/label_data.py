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
    data_dir: str = field(alias="-d")

    out_dir: str = field(alias="-o")

    dataset_name: str = field(alias="-dn", default="germeval_24")

    num_samples: int = field(alias="-n", default=100)


def main():
    args = parse(Args)
    if args.dataset_name == "germeval_24":
        dataset = load_dataset("json", data_files=args.data_dir)
        SEXISM_LEVELS = {
            "0-Kein": 0,
            "1-Gering": 1,
            "2-Vorhanden": 2,
            "3-Stark": 3,
            "4-Extrem": 4,
        }
        logger.info(SEXISM_LEVELS.keys())
    else:
        column_names = ["url", "text", "relevance", "sentiment", "aspect"]
        dataset = load_dataset(
            "csv",
            data_files="data/dev_v1.4.tsv",
            delimiter="\t",
            column_names=column_names,
        )
        logger.info("Sentiment: positive: 2; neutral: 1; negative: 0")
    logger.info(f"Loaded dataset with {len(dataset['train'])} examples")

    # Show possible classes to the user
    prediction = []
    labels = []
    correct = 0
    for i in range(args.num_samples):
        # Print text
        print(dataset["train"][i]["text"])
        # # Take user input for the label
        prediction.append(int(input("Enter label: ")))
        # Check if the label is correct
        if args.dataset_name == "germeval_24":
            votes = [SEXISM_LEVELS[annotation["label"]] for annotation in dataset["train"][i]["annotations"]]
            label = max(set(votes), key=votes.count)
            labels.append(label)
            print(label)
        else:
            sentiment = dataset["train"][i]["sentiment"]
            print(sentiment)
            if sentiment == "positive":
                label = 2
            elif sentiment == "neutral":
                label = 1
            else:
                label = 0
            labels.append(label)
            print(label)

        if prediction[i] == label:
            correct += 1
        print()

    print(f"Accuracy: {correct/args.num_samples}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_dir / "predictions.jsonl", "w") as writer:
        for i in range(args.num_samples):
            writer.write({"text": dataset["train"][i]["text"], "label": labels[i], "prediction": prediction[i]})


if __name__ == "__main__":
    main()
