#!/bin/bash
#SBATCH --job-name=cult_hybrid
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm/slurm_out/download_cult_hybrid.out
#SBATCH --error=slurm/slurm_err/download_cult_hybrid.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

python src/data_download.py --config cfgs/data_configs/cult_hybrid.yml
python src/data_download.py --config cfgs/data_configs/cult_hybrid.yml --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_hybrid_sorted --strategy sorted