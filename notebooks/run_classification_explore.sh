#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=1

# Load conda
source /home/abdullah.zubair/software/src/myconda

# Activate the conda environment
conda activate model4

# Navigate to the directory where the Python script is located
cd /work/soghigian_lab/abdullah.zubair/novel-species-detection/notebooks

# Run the Python script
python classification-explore.py
