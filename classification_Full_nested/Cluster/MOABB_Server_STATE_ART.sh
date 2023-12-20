#!/usr/bin/env bash

# Set up the enviroment and the compiler
set -e
. /home/icarrara/miniconda3/etc/profile.d/conda.sh
conda activate env_conda_igor_PyTorch

# Launch the script ($1 meaning that we have only a list of parameter)
    python3 /home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/$1/$2/$3 $4

# Deactivate the conda Enviroment
conda deactivate