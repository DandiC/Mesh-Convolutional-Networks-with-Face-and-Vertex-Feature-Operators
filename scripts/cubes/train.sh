#!/bin/bash -l

#SBATCH --job-name=Cubes_Original
#SBATCH --output=/home/dpere013/MeshCNN/outputs/cubes_original.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dpere013@odu.edu

enable_lmod

module load python/3.6
module load cuda/9.2
module load pytorch/1.0

## run the training
python train.py \
--dataroot datasets/cubes \
--name cubes_original \
--ncf 64 128 256 256 \
--pool_res 600 450 300 210 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
