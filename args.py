import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_parsing import field, list_field


@dataclass(kw_only=True)
class TrainingArgs:
    """
    Argument class for use with simple_parsing that handles the basics of most LLM training scripts. Subclass this to add more arguments.
    """

    data_dir: Path = field(alias="-d")

    hf_model_name: str = field(default="mosaicml/mosaic-bert-base", alias="--model")
    "HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise."

    from_scratch: bool = field(default=True)
    "Do not use pre-trained weights to initialize the model."

    saved_checkpoint_path: str | None = field(default=None, alias="--checkpoint")
    "Path to a saved pytorch-lightning checkpoint. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B."

    resume: bool = False

    train_files: str = field(default="train.jsonl")
    "Name of the training file."

    val_files: str = field(default="dev.jsonl")
    "Name of the validation file."

    dataset_yml: str = field(default="dataset_info.yml")
    "Path to a dataset yml file. This overrides the train_files and val_files, you only need to specify this. Specify use_train_val_info to use the train and val files."

    tokenizer_path: str | None = field(default="tokenizer")
    "Path to a saved tokenizer to switch the vocabulary. If None, use the hf_model_name."

    finetune_cfgs_after_training: str | None = field(default=None)
    "Finetune the model on all three datasets after training. Split by comma."

    save_dir: str = field(default="/hpi/fs00/share/fg-demelo/efficient-bert-pretraining/logs/")
    "Directory to save the model checkpoints."

    repo_id: str = field(default=None)
    "If specified we do not train the model, but just load the checkpoint (saved_checkpoint_path needs to be set) and upload the model to the hub. This is the ID of the repository to upload the model to"

    private: bool = field(default=False)
    "Whether to make the model private or public. Only used if repo_id is specified."

    ###############################
    ##### Training constants ######
    ###############################

    base_unit: Literal["samples", "tokens", "optimizer-steps", "iters", "epochs"] = field(default="optimizer-steps")
    "Unit of all training constants. They will be converted to optimizer_steps in __post_init__."

    training_goal: int = field(default=50_000)
    eval_interval: float = field(default=0.1)
    "Interval between evaluations. If < 1, use as percentage of training_goal. If epoachs is set we evaluate after each epoch."

    eval_samples: int = field(default=-1)
    "Number of samples on the val dataset during evaluation. If -1, use full val dataset."

    save_interval: int | float = field(default=0.1)
    "Interval between model checkpoints. If < 1, use as percentage of training_goal."

    warmup_period: float = field(default=0.06)
    "Length of lr warmup. If < 1, use as percentage of training_goal."

    lr_decay_period: int = field(default=-1)
    "If -1, decay until end of training."

    ###########################
    ##### Hyperparameters #####
    ###########################
    block_size: int = field(default=512)
    "The sequence length of samples."

    learning_rate: float = field(default=3e-4)
    batch_size: int = field(default=128, alias="-b")
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-6
    grad_clip: float = field(default=1.0)
    "If -1, disable."

    lr_schedule: Literal["cosine", "linear", "constant", "cosine_with_restarts", "polynomial"] = field(default="linear")

    #######################################
    ## Hardware acceleration & precision ##
    #######################################

    accelerator: Literal["cuda", "cpu", "tpu", "mps"] = field(default="cuda")
    "Hardware accelerator to use."
    num_devices: int = field(default=1)

    distributed_strategy: Literal["ddp", "fsdp", "auto"] = field(
        default="auto",
        help="Distributed training strategy to use. If `auto`, will select automatically (no distributed strategy is used when using a single device).",
        aliases="--ds",
    )
    micro_batch_sizes: list[int] = field(default=None, alias="--mb")
    """If None, use batch_size // num_devices. This is the batch size per device, not the total batch size.
    You should tune this so that you do not get GPU RAM OOM errors. We automatically calculate the gradient accumulation steps to achieve your desired `batch_size`.
    For each dataset you can have one batch size. Specify them for the datasets in the order they are loaded."""

    eval_micro_batch_sizes: list[int] = field(default=None)
    "If 1 use micro_batch_sizes[-1] for evaluation. Else use eval_micro_batch_sizes."

    gradient_accumulation_steps: int = field(default=-1)
    "If -1, set automatically based on batch_size and micro_batch_sizes."

    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "bf16-mixed"
    compile: bool = field(default=False)
    "torch.compile model for faster training."

    workers: int = field(default=4, alias="-w")
    preprocessing_workers: int = field(default=-1, aliases="--pw")
    "Number of workers for preprocessing the datasets. If -1, use all available CPUs."

    ############################
    ###### Logging & Misc ######
    ############################

    max_time: str = field(default="00:00:00:00")
    "Specify in the format 'days:hours:minutes:seconds'. If '00:00:00:00', no time limit."

    run_name: str = field(default=None, alias="-n")
    "Run name for logging."

    seed: int | None = field(default=None)

    only_val: bool = field(default=False)
    "Only run validation."

    val_before_training: bool = field(default=True)
    "Run one validation epoch before training."

    out_dir: Path = field(default="out/")

    wandb_tags: list[str] = list_field(default=[], alias="-t")
    "Tags for wandb."

    offline: bool = field(default=False)
    "If True, don't log to wandb."

    debug: bool = field(default=False)
    "If true, wait for debugger to attach at the start of the script."

    force_deterministic: bool = field(default=False)
    "Force PyTorch operations to be deterministic. Could be slower."

    fast_dev_run: bool = field(default=False)
    "Do fast run through training and validation with reduced sizes."

    ###############################################
    ###### Efficient Bert Pretraining Params ######
    ###############################################

    task: Literal["pretraining", "sequence-classification", "question-answering"] = field(default="pretraining")

    classifier_dropout: float = field(default=0.1)

    num_labels: int = field(default=2)

    mlm_probabilities: list[float] = list_field(default=[0.15])
    "List of probabilities for masked language modeling. Specify as many as you have train datasets."

    steps_per_seq_length: float = field(default=-1)
    "If -1, do not limit the number of steps per sequence length. This is the hard limit for the number of steps per sequence length."

    reload_dataloaders_every_n_epochs: int = field(default=0)
    "If > 0, reload the dataloaders every n epochs. Needed on 1 for dataset switching."

    dataset_switching_patience: int = field(default=5)
    "Dataset switching patience. After how many epochs without improvement (dataset_switching_delta) to switch the dataset."

    dataset_switching_delta: float = field(default=0.1)

    metric_name_for_sc: Literal["germeval_B", "germeval_24"] = field(default="germeval_24")

    def __post_init__(self):
        assert self.num_devices > 0
        if self.micro_batch_sizes is None:
            # NOTE: you need to make sure that micro_batch_sizes can fit into the GPU memory
            self.micro_batch_sizes = self.batch_size // self.num_devices
            assert self.batch_size % self.num_devices == 0

        self.iter_batch_size = self.micro_batch_sizes[-1] * self.num_devices

        if self.base_unit != "epochs":
            if self.eval_interval < 1:
                self.eval_interval = int(self.eval_interval * self.training_goal)
            if self.save_interval < 1:
                self.save_interval = int(self.save_interval * self.training_goal)
            if self.warmup_period < 1:
                self.warmup_period = int(self.warmup_period * self.training_goal)
            if self.lr_decay_period == -1:
                self.lr_decay_period = self.training_goal
            elif self.lr_decay_period < 1:
                self.lr_decay_period = int(self.lr_decay_period * self.training_goal)

        assert self.batch_size % self.micro_batch_sizes[-1] == 0
        if self.gradient_accumulation_steps == -1:
            self.gradient_accumulation_steps = self.batch_size // self.iter_batch_size
        assert self.gradient_accumulation_steps > 0
        assert self.batch_size == self.micro_batch_sizes[-1] * self.num_devices * self.gradient_accumulation_steps

        if self.tokenizer_path is None:
            self.tokenizer_path = self.hf_model_name
            assert self.hf_model_name is not None

        # Calculate training constants
        if self.base_unit == "samples":
            UNITS_PER_STEP = self.batch_size
        elif self.base_unit == "tokens":
            assert self.block_size is not None, "block_size must be set if base_unit is tokens"
            UNITS_PER_STEP = self.batch_size * self.block_size
        elif self.base_unit == "optimizer-steps":
            UNITS_PER_STEP = 1
        elif self.base_unit == "iters":
            UNITS_PER_STEP = self.gradient_accumulation_steps
        # We treat epochs in the backend as optimizer steps by calculating how many steps are in an epoch in the trainer
        elif self.base_unit == "epochs":
            UNITS_PER_STEP = None
        else:
            raise ValueError(f"Unknown training goal unit: {self.base_unit}")

        if UNITS_PER_STEP:
            self.training_goal = int(self.training_goal / UNITS_PER_STEP)
            self.eval_interval = int(self.eval_interval / UNITS_PER_STEP)
            self.save_interval = int(self.save_interval / UNITS_PER_STEP)
            self.warmup_period = int(self.warmup_period / UNITS_PER_STEP)
            self.lr_decay_period = int(self.lr_decay_period / UNITS_PER_STEP)

        if self.preprocessing_workers == -1:
            # Set to all available CPUs, handle SLURM case when only some CPUs are available to the job
            self.preprocessing_workers = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))

        self.train_files = self.train_files.split(",")
        self.val_files = self.val_files.split(",")
        self.use_n_train_datasets = len(self.train_files)
        self.use_n_val_datasets = len(self.val_files)

        self.finetune_cfgs_after_training = (
            self.finetune_cfgs_after_training.split(",") if self.finetune_cfgs_after_training else None
        )

        self.eval_micro_batch_sizes = (
            self.eval_micro_batch_sizes if self.eval_micro_batch_sizes else [self.micro_batch_sizes[-1]]
        )

        if self.max_time == "00:00:00:00":
            self.max_time = None

        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.weight_decay = float(self.weight_decay)
        self.beta1 = float(self.beta1)
        self.beta2 = float(self.beta2)
        self.epsilon = float(self.epsilon)

        if self.repo_id:
            assert self.saved_checkpoint_path is not None, "You need to specify a saved checkpoint to upload a model."

    def update_from_dict(self, values_dict):
        # Update class variables with values from the dictionary
        for key, value in values_dict.items():
            setattr(self, key, value)
