#!/bin/bash

#SBATCH -J gpu_conv_benchmark
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -A ASC23018
#SBATCH -o gpu_convolution.%j.out
#SBATCH -e gpu_convolution.%j.err

module load cuda

cd $SLURM_SUBMIT_DIR

make clean
make

./conv
