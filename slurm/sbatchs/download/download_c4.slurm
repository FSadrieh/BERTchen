#!/bin/bash
#SBATCH --job-name=c4
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/slurm_out/download_c4.out
#SBATCH --error=slurm/slurm_err/download_c4.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

python src/data_download.py --dataset c4 -o /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/c4 --only_download --processes 8 --pre_discard_factor 0.99
python src/data_download.py --config cfgs/data_configs/c4_baseline.yml --pre_discard_factor 0.75