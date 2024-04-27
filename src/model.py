import lightning as L
from print_on_steroids import logger
from torch.optim import AdamW
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.optimization import get_scheduler
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

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
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)

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

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state #TODO: equivalent to outputs[0]?
        prediction_scores = self.head(sequence_output)
        loss = self.loss_function(prediction_scores.view(-1, self.model.config.vocab_size), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(**batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(**batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.model.named_parameters() + self.head.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        scheduler_name = self.lr_schedule
        if scheduler_name == "constant" and self.warmup_period > 0:
            scheduler_name += "_with_warmup"
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            num_warmup_steps=int(self.warmup_period),
            num_training_steps=self.trainer.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class FinetuneBERT(L.LightningModule):
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
        task_type: str,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        
        self.model = model
        if task_type == "sequence-classification":
            self.head = SequenceClsHead(model.config.hidden_size, num_labels, classifier_dropout)
        else:
            self.head = TokenClsHead(model.config.hidden_size, num_labels, classifier_dropout)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        output = self.model(input_ids, attention_mask, token_type_ids=token_type_ids)
        return self.head(output, labels)

    def training_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.head.cls.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        scheduler_name = self.lr_schedule
        if scheduler_name == "constant" and self.warmup_period > 0:
            scheduler_name += "_with_warmup"
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            num_warmup_steps=int(self.warmup_period),
            num_training_steps=self.trainer.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class SequenceClsHead(nn.Module):
    def __init__(self, hidden_size, num_labels, classifier_dropout):
        super().__init__()
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = nn.Linear(hidden_size, num_labels)
        self.loss_function = nn.CrossEntropyLoss()

        self.num_labels = num_labels

    def forward(self, sequence_output: BaseModelOutputWithPoolingAndCrossAttentions, labels):
        sequence_output = self.dropout(sequence_output.pooler_output)
        logits = self.cls(sequence_output)
        loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class TokenClsHead(nn.Module):
    def __init__(self, hidden_size, num_labels, classifier_dropout):
        super().__init__()
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = nn.Linear(hidden_size, num_labels)
        self.loss_function = nn.CrossEntropyLoss()

        self.num_labels = num_labels

    def forward(self, sequence_output: BaseModelOutputWithPoolingAndCrossAttentions, labels):
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output.last_hidden_state)
        loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

