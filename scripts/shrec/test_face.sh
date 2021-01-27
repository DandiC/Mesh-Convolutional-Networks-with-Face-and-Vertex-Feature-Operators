#!/bin/bash -l

#SBATCH --job-name=Test_SHREC
#SBATCH --output=/home/dpere013/MeshCNN/outputs/shrec.txt
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dpere013@odu.edu

enable_lmod

module load python/3.6
module load cuda/9.2
module load pytorch/1.0

## run the test and export collapses
python test.py \
--dataroot datasets/shrec_16 \
--name shrec16_non_symetric_neighbors \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids -1 \
