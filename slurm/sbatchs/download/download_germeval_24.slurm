#!/bin/bash
#SBATCH --job-name=germeval_24_down
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm/slurm_out/download_germeval_24.out
#SBATCH --error=slurm/slurm_err/download_germeval_24.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

python src/data_download.py --config cfgs/data_configs/germeval_24.yml #Default tok
python src/data_download.py --config cfgs/data_configs/germeval_24.yml --tokenizer deepset/gbert-base --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_24/gBERT_tok --truncate True # For the gBERT tokenizer
python src/data_download.py --config cfgs/data_configs/germeval_24.yml --tokenizer google-bert/bert-base-uncased --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germeval_24/mosaicBERT_tok # For the mosaicBERT tokenizer