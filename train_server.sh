#!/bin/bash

#####################################
#       Configure Experiment        #
#####################################
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --job-name meniskus


source /scratch/htc/dluedke/meniscaltear/bin/activate
cd /scratch/htc/dluedke/meniscaltears
export PYTHONPATH=.


# Launch training.
exec python3 main.py "$@" #-cp ".config/" -cn config_cropped
