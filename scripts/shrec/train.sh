#!/bin/bash -l

#SBATCH --job-name=shrec
#SBATCH --output=outputs/face_pool2_shrec.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dpere013@odu.edu

enable_lmod

module load python/3.6
module load cuda/9.2
module load pytorch/1.0
module load py-wandb

## run the training
python train.py \
--dataroot datasets/shrec_16 \
--name shrec16 \
--ncf 64 128 256 256 \
--pool_res 500 400 300 200 120 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--gpu_ids 0 \
--ninput_features 500 \
--feat_from face
