#!/bin/bash
#SBATCH --job-name=short_cult_baseline
#SBATCH -A demelo-mpss2024gd1
#SBATCH --partition sorcery
#SBATCH -C GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=1-10:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm/slurm_out/cult_baseline_short.out
#SBATCH --error=slurm/slurm_err/cult_baseline_short.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

# Pretrain with old tokenizer
python train_interface.py --config cfgs/baseline_short.yml --run_name default_tok_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_original_tok_baseline_512 --micro_batch_sizes 64 --batch_size 1024 --tokenizer google-bert/bert-base-uncased --finetune_cfgs_after_training: None

#Finetune baselines
python train_interface.py --config cfgs/germeval_B.yml --hf_model_name deepset/gbert-base --tokenizer_path deepset/gbert-base --run_name gBERT_germeval_B --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_B/gBERT_tok
python train_interface.py --config cfgs/germeval_24.yml --hf_model_name deepset/gbert-base --tokenizer_path deepset/gbert-base --run_name gBERT_germeval_24 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_24/gBERT_tok
python train_interface.py --config cfgs/germanquad.yml --hf_model_name deepset/gbert-base --tokenizer_path deepset/gbert-base --run_name gBERT_germanquad --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germanquad/gBERT_tok

python train_interface.py --config cfgs/germeval_B.yml --hf_model_name mosaicml/mosaic-bert-base --tokenizer_path google-bert/bert-base-uncased --run_name mosaicBERT_germeval_B --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_B/mosaicBERT_tok
python train_interface.py --config cfgs/germeval_24.yml --hf_model_name mosaicml/mosaic-bert-base --tokenizer_path google-bert/bert-base-uncased --run_name mosaicBERT_germeval_24 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_24/mosaicBERT_tok
python train_interface.py --config cfgs/germanquad.yml --hf_model_name mosaicml/mosaic-bert-base --tokenizer_path google-bert/bert-base-uncased --run_name mosaicBERT_germanquad --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germanquad/mosaicBERT_tok

python train_interface.py --config cfgs/germeval_B.yml --tokenizer_path google-bert/bert-base-uncased --run_name original_tok_germeval_B --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_B/mosaicBERT_tok --checkpoint /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/logs/bert-pretraining/default_tok_baseline_512/checkpoints/last_model_ckpt.ckpt
python train_interface.py --config cfgs/germeval_24.yml --tokenizer_path google-bert/bert-base-uncased --run_name original_tok_germeval_24 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_24/mosaicBERT_tok --checkpoint /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/logs/bert-pretraining/default_tok_baseline_512/checkpoints/last_model_ckpt.ckpt
python train_interface.py --config cfgs/germanquad.yml --tokenizer_path google-bert/bert-base-uncased --run_name original_tok_germanquad --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germanquad/mosaicBERT_tok --checkpoint /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/logs/bert-pretraining/default_tok_baseline_512/checkpoints/last_model_ckpt.ckpt