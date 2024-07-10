from pathlib import Path

import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.plugins.environments import LightningEnvironment
from print_on_steroids import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import yaml
import math
from src.custom_wandb_logger import CustomWandbLogger

from args import TrainingArgs
from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank

from src.data_loading import LMDataModule
from src.helpers import ProgressMetricCallback, check_for_wandb_checkpoint_and_download_if_necessary
from src.model import PretrainBERT, QABERT, SCBERT
from src.custom_train_loop.trainer import CustomTrainer


def train(args: TrainingArgs, wandb_logger: CustomWandbLogger, IS_ON_SLURM: bool = False):
    ########### CUDA checks ###########
    current_process_rank = get_rank()
    logger.config(rank=current_process_rank, print_rank0_only=True)

    # Specify the train and val files
    if args.dataset_yml != "use_train_val_info":
        yml_file_path = str(args.data_dir / args.dataset_yml)
        with open(yml_file_path, "r") as f:
            dataset_yml = yaml.safe_load(f)
        train_files, val_files = [], []
        val_set_to_seq_len = {}
        count = 0
        for key in dataset_yml.keys():
            if key.startswith("train"):
                train_files.append(key)
            elif key.startswith("dev"):
                val_files.append(key)
                val_set_to_seq_len[count] = f"val/loss_{dataset_yml[key]['seq_len']}"
                count += 1
        monitor_loss = val_set_to_seq_len[len(val_files) - 1]
        args.use_n_train_datasets = len(train_files)
        args.use_n_val_datasets = len(val_files)
    else:
        train_files = args.train_files
        val_files = args.val_files
        val_set_to_seq_len = None
        # This can be wrong. Better specify a yml_file
        monitor_loss = "val/loss" if args.use_n_val_datasets == 1 else "val/loss_512"

    assert len(train_files) == len(args.mlm_probabilities), "Number of train files must match number of mlm_probabilities"

    ################# Construct model ##############

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer_path or args.hf_model_name, use_fast=True)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    dm = LMDataModule(training_args=args, tokenizer=tokenizer, train_files=train_files, val_files=val_files)
    if args.base_unit == "epochs":
        dm.setup(None)
        steps_per_epoch = math.ceil(len(dm.train_dataset) / args.batch_size)

        max_steps = args.training_goal * steps_per_epoch
        eval_interval = steps_per_epoch
        save_interval = steps_per_epoch
        warmup_period = int(args.warmup_period * max_steps)
    else:
        max_steps = args.training_goal
        eval_interval = args.eval_interval
        save_interval = args.save_interval
        warmup_period = args.warmup_period

    # Resume from checkpoint if specified
    model_args = dict(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        lr_schedule=args.lr_schedule,
        warmup_period=warmup_period,
        eval_interval=args.eval_interval,
        epsilon=args.epsilon,
    )
    if args.saved_checkpoint_path:
        args.saved_checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.saved_checkpoint_path, wandb_logger.experiment
        )

        if args.resume:  # load weights, optimizer states, scheduler state, ...\. Note we do not support resuming finetuning.
            model = PretrainBERT.load_from_checkpoint(args.saved_checkpoint_path, save_hyperparameters=False)
            # we will resume via trainer.fit(ckpt_path=...)
        else:  # load only weights
            model = PretrainBERT(
                **model_args,
                model_name_or_path=args.hf_model_name,
                from_scratch=args.from_scratch,
                tokenizer_vocab_size=tokenizer.vocab_size,
                use_n_val_datasets=args.use_n_val_datasets,
                val_set_to_seq_len=val_set_to_seq_len,
            )
            torch_load = torch.load(args.saved_checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = PretrainBERT(
            **model_args,
            model_name_or_path=args.hf_model_name,
            from_scratch=args.from_scratch,
            tokenizer_vocab_size=tokenizer.vocab_size,
            use_n_val_datasets=args.use_n_val_datasets,
            val_set_to_seq_len=val_set_to_seq_len,
        )

    if args.task == "question-answering":
        model = QABERT(
            model=model.model,
            **model_args,
            num_labels=args.num_labels,
        )
    elif args.task == "sequence-classification":
        model = SCBERT(
            model=model.model,
            **model_args,
            num_labels=args.num_labels,
            classifier_dropout=args.classifier_dropout,
            metric_name=args.metric_name_for_sc,
        )

    if not args.resume:
        pretrained_vocab_size = model.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != pretrained_vocab_size:
            logger.warning(f"Resizing embedding size from {pretrained_vocab_size} to match tokenizer ({len(tokenizer)}).")
            model.model.resize_token_embeddings(len(tokenizer))

    wandb_logger.watch(model, log="all", log_freq=500, log_graph=False)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
        model = torch.compile(model)

    #################### Upload model if specified #################

    if args.repo_id:
        logger.info("Uploading model to Hugging Face Model Hub...")
        model.model.push_to_hub(repo_id=args.repo_id, private=args.private, token=True, safe_serialization=True)
        logger.success("Model uploaded successfully! Will not train.")
        exit(0)

    #################### Construct trainer #################

    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(cleanup_local=True, cleanup_online=False, size_limit=20)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}bert-pretraining/{args.run_name}/checkpoints/",
        filename="snap-{step}-samples-{progress/samples}-{progress/tokens}-loss-{val/loss:.2f}",
        monitor=monitor_loss,
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=int(save_interval),
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss, min_delta=args.early_stopping_delta, patience=args.early_stopping_patience
    )
    callbacks = [
        wandb_disk_cleanup_callback,
        lr_monitor,
    ]

    if args.task == "pretraining":
        callbacks.append(ProgressMetricCallback())
        callbacks.append(checkpoint_callback)
        # callbacks.append(early_stopping_callback)
    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    plugins = None
    if IS_ON_SLURM:
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins = [LightningEnvironment()]

    # lightning wants val_check_interval in num forward passes (iters) not num optimization steps
    val_frequency_in_iters = eval_interval * args.gradient_accumulation_steps

    trainer_dict = dict(
        devices=args.num_devices,
        accelerator=args.accelerator,
        strategy=args.distributed_strategy,
        logger=wandb_logger,
        deterministic=args.force_deterministic,
        # callbacks=callbacks,
        plugins=plugins,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=args.fast_dev_run,
        limit_val_batches=(None if args.eval_samples == -1 else (args.eval_samples // args.eval_micro_batch_sizes[-1])),
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
        limit_train_batches=None if args.steps_per_seq_length == -1 else args.steps_per_seq_length,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        max_time=args.max_time,
        max_steps=max_steps,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        val_check_interval=val_frequency_in_iters,
    )

    # We only want dataset switching for pretraining
    if args.task == "pretraining":
        pretraining_args = dict(
            monitor=monitor_loss,
            min_delta=args.dataset_switching_delta,
            patience=args.dataset_switching_patience,
            num_datasets=args.use_n_train_datasets,
            mode="min",
        )
        trainer = CustomTrainer(**trainer_dict, **pretraining_args)
    else:
        trainer = Trainer(**trainer_dict)

    if current_process_rank == 0:
        logger.info(
            f"Total optimizer steps: {args.training_goal} | "
            f"LR warmup steps: {warmup_period} | "
            f"Validation Frequency: {args.eval_interval} | "
            f"Model Log Frequency: {save_interval} | "
            f"Effective batch size: {args.batch_size} | "
            f"Micro batch size (per device and forward pass): {args.eval_micro_batch_sizes[-1]} | "
            f"Gradient accumulation steps: {args.gradient_accumulation_steps} | "
        )

    ########### Start val & train loop ###########
    if args.val_before_training and not args.resume:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        if args.only_val:
            exit(0)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, dm, ckpt_path=args.saved_checkpoint_path if args.resume else None)
    if trainer.interrupted and IS_ON_SLURM:
        logger.error(
            "Detected keyboard interrupt, not trying to save latest checkpoint right now because we detected SLURM and do not want to drain the node..."
        )
    else:
        if trainer.interrupted:
            logger.warning("Detected keyboard interrupt, trying to save latest checkpoint...")
        else:
            logger.success("Fit complete, starting validation...")
            val_results = trainer.validate(model, dm)

        if current_process_rank == 0:
            logger.info("Trying to save checkpoint....")

            save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
            trainer.save_checkpoint(save_path)

            logger.info("Collecting PL checkpoint for wandb...")
            artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
            artifact.add_file(save_path, name="model.ckpt")

            logger.info("Pushing to wandb...")
            aliases = ["train_end", "latest"]
            wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

            logger.success("Saving finished!")

        return val_results
