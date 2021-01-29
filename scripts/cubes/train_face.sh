#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/cubes \
--name cubes_face \
--ncf 128 256 256 512 \
--pool_res 400 300 200 140 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--gpu_ids 0 \
--ninput_features 500 \
--feat_from face \
--clean_data
