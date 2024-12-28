#!/bin/bash
#SBATCH --mem=60GB
#SBATCH --job-name=8TFT8_TeLU
#SBATCH --gpus-per-task=4
#SBATCH --gpus=4
#SBATCH -o 8TFT8_TeLU.out
#SBATCH -e 8TFT8_TeLU.err
#SBATCH -p Extended
C=configs/whitespaces_telu_1.yaml GPUS=4 bash scripts/run_exp.sh

