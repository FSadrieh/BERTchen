from print_on_steroids import graceful_exceptions, logger
from simple_parsing import parse, parse_known_args
import wandb
import yaml
from pathlib import Path
import torch
from lightning import seed_everything
import dataclasses
import os
from lightning.pytorch.plugins.environments import SLURMEnvironment

from dlib import get_rank, log_slurm_info
from src.utils import wait_for_debugger
from src.helpers import check_checkpoint_path_for_wandb
from src.custom_wandb_logger import CustomWandbLogger
import train
from args import TrainingArgs

WANDB_PROJECT = "bert-pretraining"
WANDB_ENTITY = "raphael-team"


def main(current_process_rank):
    args = parse(TrainingArgs, add_config_path_arg=True)
    if args.debug:
        wait_for_debugger()

    wandb_logger, args, IS_ON_SLURM = create_wandb_logger(args, current_process_rank)

    val_results = train.train(args, wandb_logger, IS_ON_SLURM)
    logger.info(val_results)
    run_name = args.run_name
    if args.finetune_cfgs_after_training:
        finetune_results = []
        for cfg in args.finetune_cfgs_after_training:
            finetune_args = {}
            # We do not want to use the --config argument for the finetuning therefore we need to load the config file manually
            with open(cfg, "r") as stream:
                try:
                    finetune_args = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    logger.error(exc)

            finetune_args["data_dir"] = Path(finetune_args["data_dir"])
            finetune_args["run_name"] = run_name + "_" + cfg.split("/")[-1].split(".")[0]
            finetune_args[
                "saved_checkpoint_path"
            ] = f"{args.save_dir}bert-pretraining/{run_name}/checkpoints/last_model_ckpt.ckpt"

            new_args = TrainingArgs(**finetune_args)
            logger.info(f"Finetuning {finetune_args['run_name']}")
            finetune_result = train.train(new_args, wandb_logger)
            finetune_results.append(f"{new_args.run_name},{finetune_result}\n")
        for line in finetune_results:
            with open("results.csv", "a") as fd:
                fd.write(line)
            logger.info(line)


def create_wandb_logger(args, current_process_rank):
    if args.accelerator == "cuda":
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus > args.num_devices:
            logger.warning(
                f"Requested {args.num_devices} GPUs but {num_available_gpus} are available. Using first {args.num_devices} GPUs. You should set CUDA_VISIBLE_DEVICES or the docker --gpus flag to the desired GPU ids.",
            )
        if not torch.cuda.is_available():
            logger.error("CUDA is not available, you should change the accelerator with --accelerator cpu|tpu|mps.")
            exit(1)
    args.seed = seed_everything(workers=True, seed=args.seed)

    ############# Construct W&B Logger ##############
    if args.offline or args.fast_dev_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb_extra_args = dict(name=args.run_name)
    if args.saved_checkpoint_path and args.resume and check_checkpoint_path_for_wandb(args.saved_checkpoint_path):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(id=check_checkpoint_path_for_wandb(args.saved_checkpoint_path), resume="must")  # resume W&B run
    wandb_logger = CustomWandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model=True,
        tags=args.wandb_tags,
        save_dir="logs/",
        **wandb_extra_args,
    )

    if current_process_rank == 0:
        logger.info(args)
    if current_process_rank == 0 and not args.resume and not args.offline:
        if args.run_name is None:
            logger.warning("No run name specified with `--run_name`. Using W&B default (randomly generated name).")
    IS_ON_SLURM = SLURMEnvironment.detect()
    if IS_ON_SLURM and current_process_rank == 0:
        log_slurm_info()

    wandb_logger.log_hyperparams(dataclasses.asdict(args))
    return wandb_logger, args, IS_ON_SLURM


def sweep(config_path):
    # For sweeps we get the params from the sweep config
    wandb.init()
    args, __ = parse_known_args(TrainingArgs, config_path=config_path)
    args.update_from_dict(wandb.config)
    wandb_logger, args, IS_ON_SLURM = create_wandb_logger(args, 0)
    train.train(args, wandb_logger, IS_ON_SLURM)


if __name__ == "__main__":
    current_process_rank = get_rank()
    with graceful_exceptions(extra_message=f"Rank: {current_process_rank}"):
        main(current_process_rank)
