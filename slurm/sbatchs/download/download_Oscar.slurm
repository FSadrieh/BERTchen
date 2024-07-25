#!/bin/bash
#SBATCH --job-name=cult_long
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm/slurm_out/download_cult4.out
#SBATCH --error=slurm/slurm_err/download_cult4.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

python src/data_download.py --dataset oscar2023 -o /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/Oscar_20_percent --only_download --processes 8 --pre_discard_factor 0.8
python src/data_download.py --config cfgs/data_configs/oscar.yml
