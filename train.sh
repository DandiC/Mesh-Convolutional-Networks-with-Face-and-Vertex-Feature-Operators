#!/bin/bash -l

#SBATCH --job-name=wandbTest 
#SBATCH --output=outputs/wandbtest.txt 
#SBATCH --partition=timed-gpu
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=dpere013@odu.edu

enable_lmod

module load python/3.6
module load cuda/9.2
module load pytorch/1.0
module load wandb


python train.py \
--dataroot datasets/shrec_16 \
--name test \
--ncf 64 128 256 256 \
--pool_res 600 450 300 210 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
