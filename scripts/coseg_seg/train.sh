#!/bin/bash -l

#SBATCH --job-name=Coseg_Original
#SBATCH --output=/home/dpere013/MeshCNN/outputs/coseg_original.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dpere013@odu.edu

enable_lmod

module load python/3.6
module load cuda/9.2
module load pytorch/1.0

## run the training
python train.py --dataroot datasets/coseg_aliens --name coseg_aliens_original --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 --ninput_edges 2280 --pool_res 1800 1350 600 --resblocks 3 --lr 0.001 --batch_size 12 --num_aug 20 --slide_verts 0.2
python train.py --dataroot datasets/coseg_vases --name coseg_vases_original --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 --ninput_edges 1500 --pool_res 1050 600 300 --resblocks 3 --lr 0.001 --batch_size 12 --num_aug 20
