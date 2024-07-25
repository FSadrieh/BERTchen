#!/bin/bash
#SBATCH --job-name=germanquad_down
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm/slurm_out/download_germanquad.out
#SBATCH --error=slurm/slurm_err/download_germanquad.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03


python src/data_download.py --config cfgs/data_configs/germanquad.yml # For the default tokenizer
python src/data_download.py --config cfgs/data_configs/germanquad.yml --tokenizer deepset/gbert-base --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germanquad/gBERT_tok --truncate True # For the gBERT tokenizer
python src/data_download.py --config cfgs/data_configs/germanquad.yml --tokenizer google-bert/bert-base-uncased --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/germanquad/mosaicBERT_tok # For the mosaicBERT tokenizer