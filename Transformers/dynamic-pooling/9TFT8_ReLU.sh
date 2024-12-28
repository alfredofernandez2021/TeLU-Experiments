#!/bin/bash
#SBATCH --mem=60GB
#SBATCH --job-name=9TFT8_ReLU
#SBATCH --gpus-per-task=4
#SBATCH --gpus=4
#SBATCH -o 9TFT8_ReLU.out
#SBATCH -e 9TFT8_ReLU.err
#SBATCH -p Extended
C=configs/whitespaces_relu_3.yaml GPUS=4 bash scripts/run_exp.sh

