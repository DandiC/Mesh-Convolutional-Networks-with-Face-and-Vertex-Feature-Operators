#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/cubes \
--name cubes_vertex \
--ncf 128 256 256 512 \
--pool_res 200 150 100 70 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--ninput_features 252 \
--feat_from point \
--clean_data

