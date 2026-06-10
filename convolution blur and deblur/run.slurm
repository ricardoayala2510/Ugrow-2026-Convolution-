#!/bin/bash

#SBATCH -J convolution
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A ASC23018
#SBATCH -o convolution.%j.out
#SBATCH -e convolution.%j.err

module load cuda

cd $SLURM_SUBMIT_DIR

echo "Running on:"
hostname

echo "Checking GPU:"
nvidia-smi

echo "Starting program..."
./conv

echo "Done."
