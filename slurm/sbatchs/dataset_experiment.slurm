#!/bin/bash
#SBATCH --job-name=different_datasets
#SBATCH -A demelo-mpss2024gd1
#SBATCH --partition sorcery
#SBATCH -C GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --time=1-10:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm/slurm_out/different_datasets.out
#SBATCH --error=slurm/slurm_err/different_datasets.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ld

python train_interface.py --config cfgs/baseline_short.yml --run_name c4_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/c4/c4_baseline_512 --micro_batch_sizes 64 --batch_size 1024
python train_interface.py --config cfgs/baseline_short.yml --run_name Oscar_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/Oscar_20_percent/normal --micro_batch_sizes 64 --batch_size 1024
python train_interface.py --config cfgs/baseline_short.yml --run_name wiki_books_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/wiki_books/wiki_books_baseline_512 --micro_batch_sizes 64 --batch_size 1024
python train_interface.py --config cfgs/baseline_short.yml --run_name wiki_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/wikipedia/normal --micro_batch_sizes 64 --batch_size 1024
python train_interface.py --config cfgs/baseline_short.yml --run_name books_baseline_512 --data_dir /hpi/fs00/share/fg-demelo/efficient-bert-pretraining/data/books/normal --micro_batch_sizes 64 --batch_size 1024