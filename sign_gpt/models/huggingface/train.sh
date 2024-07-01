#!/bin/bash

#SBATCH --job-name=train-llama
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=train-%j.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate sign_gpt

python -m sign_gpt.models.huggingface.train_lora \
  --output-dir="/shares/iict-sp2.ebling.cl.uzh/amoryo/checkpoints/sign-gpt/llama"
