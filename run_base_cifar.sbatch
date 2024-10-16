#!/bin/bash

#SBATCH --job-name=cl
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=o_base_cifar_cap.job
#SBATCH --error=e_base_cifar_cap.job

# Load the conda environment
source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_new_env

# Run the Python script
python 'train_cifar_base.py' --model cnn --dataset cifar10 --batch_size 128 --epochs 90 --learning_rate 0.0001 --explainer compute_input_x_gradient --metric sufficiency infidelity sensitivity

