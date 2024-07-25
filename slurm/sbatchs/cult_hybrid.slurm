#!/bin/bash
#SBATCH --job-name=hybrid_cult
#SBATCH -A demelo-mpss2024gd1
#SBATCH --partition sorcery
#SBATCH -C GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm/slurm_out/hybrid_cult.out
#SBATCH --error=slurm/slurm_err/hybrid_cult.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

# Default it runs naive hybrid approach for CulturaX
python train_interface.py --config cfgs/hybrid.yml
python train_interface.py --config cfgs/hybrid.yml --run_name CulturaX_hybrid_sorted --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/CulturaX_2_percent/CulturaX_hybrid_sorted
python train_interface.py --config cfgs/hybrid.yml --run_name CulturaX_hybrid_mask_descending --mlm_probabilities 0.3 0.25 0.2 0.15
python train_interface.py --config cfgs/hybrid.yml --run_name CulturaX_hybrid_mask_descending_double --mlm_probabilities 0.6 0.5 0.4 0.3