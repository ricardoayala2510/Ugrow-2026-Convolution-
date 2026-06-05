#!/bin/bash

#SBATCH -J convolution
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -A YOUR_ALLOCATION

module load cuda

./conv

