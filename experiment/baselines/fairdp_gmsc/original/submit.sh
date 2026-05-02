#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J FairDP
#SBATCH -p datasci
#SBATCH --output=results/logs/adult_ns_0.8.out
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
module load python
conda activate torch
python main.py --mode proposed --dataset adult --batch_size 512 --lr 0.1 --epochs 50 --clip 0.04 --ns 0.8
python main.py --mode proposed --dataset adult --batch_size 512 --lr 0.1 --epochs 50 --clip 0.04 --ns 0.8
python main.py --mode proposed --dataset adult --batch_size 512 --lr 0.1 --epochs 50 --clip 0.04 --ns 0.8
python main.py --mode proposed --dataset adult --batch_size 512 --lr 0.1 --epochs 50 --clip 0.04 --ns 0.8
python main.py --mode proposed --dataset adult --batch_size 512 --lr 0.1 --epochs 50 --clip 0.04 --ns 0.8
