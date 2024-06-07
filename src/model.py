import lightning as L
import torch.nn as nn
from evaluate import load
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.modeling_utils import PreTrainedModel
import collections
import torch

from src.utils import configure_optimizer


class PretrainBERT(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        eval_interval: int,
        tokenizer_vocab_size: int,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        use_n_val_datasets: int = 1,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)

        if from_scratch:
            config.vocab_size = tokenizer_vocab_size

        if model_name_or_path.startswith("mosaic"):
            from src.mosaic_bert import BertModel, BertOnlyMLMHead

            self.model: PreTrainedModel = (
                BertModel.from_pretrained(model_name_or_path, config=config) if not from_scratch else BertModel(config=config)
            )
            self.head = BertOnlyMLMHead(config, self.model.embeddings.word_embeddings.weight)
        else:
            from transformers.models.bert.modeling_bert import BertOnlyMLMHead

            self.model: PreTrainedModel = (
                AutoModel.from_pretrained(model_name_or_path, config=config)
                if not from_scratch
                else AutoModel.from_config(config=config)
            )
            self.head = BertOnlyMLMHead(config)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon
        self.loss_function = nn.CrossEntropyLoss()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.val_dataloader_to_seq_len = {
            0: "val/loss_64" if use_n_val_datasets > 1 else "val/loss",
            1: "val/loss_128",
            2: "val/loss_256",
            3: "val/loss_512",
        }

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        last_hidden_states = self.model(input_ids, attention_mask, output_hidden_states=True)[0]
        prediction_scores = self.head(last_hidden_states)
        loss = self.loss_function(prediction_scores.view(-1, self.model.config.vocab_size), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        self.start.record()
        loss = self(**batch)
        self.end.record()
        torch.cuda.synchronize()
        self.log(
            "tokens/sec", batch["input_ids"].numel() / (self.start.elapsed_time(self.end) / 1000), on_step=True, on_epoch=False
        )
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(**batch)
        self.log(self.val_dataloader_to_seq_len[dataloader_idx], loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return configure_optimizer(
            list(self.model.named_parameters()) + list(self.head.named_parameters()),
            self.global_rank,
            self.learning_rate,
            self.weight_decay,
            self.warmup_period,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.lr_schedule,
            self.trainer,
        )


class QABERT(L.LightningModule):
    def __init__(
        self,
        model: AutoModel,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        eval_interval: int,
        num_labels: int,
        finetune_last_layer: bool,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.model = model
        self.head = QuestionAnsweringHead(model.config.hidden_size, num_labels)
        self.metric = load("squad")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon
        self.finetune_last_layer = finetune_last_layer

    def forward(self, input_ids, attention_mask, start_positions, end_positions, token_type_ids=None):
        output = self.model(input_ids, attention_mask)[0]
        return self.head(output, start_positions=start_positions, end_positions=end_positions)

    def training_step(self, batch, batch_idx):
        loss = self(**batch["input"])[0]
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, start_logits, end_logits = self(**batch["input"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        # We want to restore the predictions to the original format {'prediction_text': , 'id': }
        prediction = self.compute_prediction(start_logits, end_logits, batch["ids"], batch["context"], batch["offset_mapping"])
        # The answer has the format {'answers': {'answer_start': , 'text': }, 'id': }
        answers = batch["answers"]
        results = self.metric.compute(predictions=prediction, references=answers)
        self.log("val/f1", results["f1"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/exact_match", results["exact_match"], on_step=False, on_epoch=True, sync_dist=True)

    def compute_prediction(self, start_logits, end_logits, ids, contexts, offset_mapping, n_best=20, max_answer_length=30):
        # The code is adapted from the huggingface guide to QA https://huggingface.co/learn/nlp-course/chapter7/7
        example_to_features = collections.defaultdict(list)
        for idx, example_id in enumerate(ids):
            example_to_features[example_id].append(idx)

        predicted_answers = []
        for example_id, context in zip(ids, contexts):
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = offset_mapping[feature_index]

                start_indexes = torch.argsort(start_logit)[-n_best - 1 :].tolist()
                end_indexes = torch.argsort(end_logit)[-n_best - 1 :].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        try:
                            if offsets[start_index] is None or offsets[end_index] is None:
                                continue
                        except IndexError:
                            # This happens when we use padding
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append({"prediction_text": best_answer["text"], "id": str(example_id)})
            else:
                predicted_answers.append({"prediction_text": "", "id": str(example_id)})

        return predicted_answers

    def configure_optimizers(self):
        trainable_parameters = list(self.head.qa.named_parameters())
        if self.finetune_last_layer:
            trainable_parameters += list(self.model.encoder.layer[-1].named_parameters())
        return configure_optimizer(
            trainable_parameters,
            self.global_rank,
            self.learning_rate,
            self.weight_decay,
            self.warmup_period,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.lr_schedule,
            self.trainer,
        )


class QuestionAnsweringHead(nn.Module):
    # Similar to the BertForQuestionAnswering class in the transformers library: https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/bert/modeling_bert.py#L1774
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.qa = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, start_positions, end_positions):
        logits = self.qa(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        return loss, start_logits, end_logits


class SCBERT(L.LightningModule):
    def __init__(
        self,
        model: AutoModel,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        eval_interval: int,
        num_labels: int,
        classifier_dropout: float,
        finetune_last_layer: bool,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.model = model
        self.head = SequenceClsHead(model.config.hidden_size, num_labels, classifier_dropout)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon
        self.finetune_last_layer = finetune_last_layer

        self.metric = load("accuracy")

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        last_hidden_state = self.model(input_ids, attention_mask)[0]
        return self.head(last_hidden_state, labels, attention_mask)

    def training_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        predictions = torch.argmax(logits, axis=1)
        accuracy = self.metric.compute(predictions=predictions, references=batch["labels"])["accuracy"]
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        trainable_parameters = list(self.head.cls.named_parameters())
        if self.finetune_last_layer:
            trainable_parameters += list(self.model.encoder.layer[-1].named_parameters())
        return configure_optimizer(
            trainable_parameters,
            self.global_rank,
            self.learning_rate,
            self.weight_decay,
            self.warmup_period,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.lr_schedule,
            self.trainer,
        )


class SequenceClsHead(nn.Module):
    def __init__(self, hidden_size, num_labels, classifier_dropout):
        super().__init__()
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = nn.Linear(hidden_size, num_labels)
        self.loss_function = nn.CrossEntropyLoss()

        self.num_labels = num_labels

    def forward(self, last_hidden_state: torch.Tensor, labels, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        avg_last_hidden_state = torch.sum(last_hidden_state * expanded_attention_mask, dim=1) / torch.sum(
            attention_mask, dim=1, keepdim=True
        )
        avg_last_hidden_state = self.dropout(avg_last_hidden_state)
        logits = self.cls(avg_last_hidden_state)
        loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
