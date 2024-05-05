from transformers import DataCollatorWithPadding


class QADataCollator:
    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None):
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=padding, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        batch_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "start_positions": feature["start_positions"],
                "end_positions": feature["end_positions"],
            }
            for feature in features
        ]
        batch = self.data_collator(batch_features)
        answers = [feature["answers"] for feature in features]
        ids = [feature["id"] for feature in features]
        context = [feature["context"] for feature in features]
        offset_mapping = [feature["offset_mapping"] for feature in features]
        return {"input": batch, "answers": answers, "ids": ids, "context": context, "offset_mapping": offset_mapping}
