#!/bin/bash
#SBATCH --job-name=long_cult_baseline
#SBATCH -A demelo-mpss2024gd1
#SBATCH --partition sorcery
#SBATCH -C GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=1-10:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=frederic.sadrieh@student.hpi.de
#SBATCH --mail-type=ALL
#SBATCH --verbose
#SBATCH --output=slurm/slurm_out/cult_baseline_long.out
#SBATCH --error=slurm/slurm_err/cult_baseline_long.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

python train_interface.py --config cfgs/baseline_short.yml --run_name cult_baseline_512_long --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_baseline_512  --micro_batch_sizes 64 --batch_size 1024 --training_goal 17_000 --max_time 01:00:00:00 --workers 10
