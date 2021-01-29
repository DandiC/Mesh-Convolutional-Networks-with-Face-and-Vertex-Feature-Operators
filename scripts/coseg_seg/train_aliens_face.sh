#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/coseg_aliens \
--name aliens_face \
--arch meshunet \
--dataset_mode segmentation \
--ncf 64 128 256 256 \
--pool_res 1200 900 600 \
--norm batch \
--resblocks 3 \
--lr 0.001 \
--batch_size 12 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--gpu_ids 0 \
--ninput_features 1500 \
--feat_from face \
--clean_data
