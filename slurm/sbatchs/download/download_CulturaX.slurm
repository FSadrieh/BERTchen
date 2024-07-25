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

python src/data_download.py --dataset CulturaX -o /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent --only_download --processes 8 --pre_discard_factor 0.98
python src/data_download.py --config cfgs/data_configs/cult_baseline_64.yml
python src/data_download.py --config cfgs/data_configs/cult_baseline_128.yml
python src/data_download.py --config cfgs/data_configs/cult_baseline_256.yml
python src/data_download.py --config cfgs/data_configs/cult_baseline_512.yml
python src/data_download.py --config cfgs/data_configs/cult_baseline_512.yml --out_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_original_tok_baseline_512 --tokenizer google-bert/bert-base-uncased
