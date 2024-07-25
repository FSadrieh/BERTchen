#!/bin/bash
#SBATCH --job-name=wiki_books
#SBATCH -A demelo-student
#SBATCH --partition magic
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=50
#SBATCH --mem=300G
#SBATCH --output=slurm/slurm_out/wiki_books.out
#SBATCH --error=slurm/slurm_err/wiki_books.err
#SBATCH --exclude=fp01,fp02,cp01,cp02,cp03

python src/data_download.py --config cfgs/data_configs/wiki_books.yml
python src/data_download.py --config cfgs/data_configs/just_wiki.yml
python src/data_download.py --config cfgs/data_configs/just_books.yml