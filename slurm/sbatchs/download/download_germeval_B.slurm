#!/bin/bash
#SBATCH --job-name=germeval_B_down
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm/slurm_out/download_germeval_B.out
#SBATCH --error=slurm/slurm_err/download_germeval_B.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

python src/data_download.py --config cfgs/data_configs/germeval_B.yml # Default tok
python src/data_download.py --config cfgs/data_configs/germeval_B.yml --tokenizer deepset/gbert-base --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_B/gBERT_tok --truncate True # For the gBERT tokenizer
python src/data_download.py --config cfgs/data_configs/germeval_B.yml --tokenizer google-bert/bert-base-uncased --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_B/mosaicBERT_tok # For the mosaicBERT tokenizer