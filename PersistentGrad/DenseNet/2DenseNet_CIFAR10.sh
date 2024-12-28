#!/bin/bash
#SBATCH --mem=60GB
#SBATCH --job-name=2DenseNet_CIFAR10
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#SBATCH -o 2DenseNet_CIFAR10.out
#SBATCH -e 2DenseNet_CIFAR10.err
#SBATCH -p Quick
python 2DenseNet_CIFAR10.py

