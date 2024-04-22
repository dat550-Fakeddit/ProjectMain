#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=08:00:00
#SBATCH --job-name=6_way_CLIP
#SBATCH --output=LOGS_TRAINING/6_way_CLIP.out
 
# Activate environment
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39
conda activate bert_cnn_cuda12
PATH=~/.local/bin:$PATH
echo $PATH
# Run the Python script that uses the GPU
TOKENIZERS_PARALLELISM=false
srun python -u Train_CLIP.py