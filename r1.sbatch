#!/bin/bash

#SBATCH --job-name=cl
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=r1_eth_cap.job
#SBATCH --error=r2_eth_cap.job

# Load the conda environment
source /home/aih/gizem.mert/tools/apps/mamba/etc/profile.d/conda.sh
conda activate env

# Run the Python script
python 'train_dropout.py' --model vit --dataset bloodmnist --batch_size 128 --epochs 150 --learning_rate 0.0001 --weight_decay 0 --explainer saliency --scheduler sufficiency --metric sufficiency infidelity sensitivity

