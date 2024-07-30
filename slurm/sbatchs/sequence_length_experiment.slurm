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

# Default it runs 128 seq len. Listed in seq len decreasing order
python train_interface.py --config cfgs/baseline_short.yml --run_name cult_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_baseline_512  --micro_batch_sizes 64 --batch_size 1024
python train_interface.py --config cfgs/baseline_short.yml --run_name cult_baseline_256 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_baseline_256 --micro_batch_sizes 128 --batch_size 2048
python train_interface.py --config cfgs/baseline_short.yml
python train_interface.py --config cfgs/baseline_short.yml --run_name cult_baseline_64 --data_dir //hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_baseline_64 --micro_batch_sizes 512 --batch_size 8192
