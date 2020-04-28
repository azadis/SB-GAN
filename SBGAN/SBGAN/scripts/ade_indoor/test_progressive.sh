#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================

# ade_indoor
CUDA_VISIBLE_DEVICES=1 python SBGAN/trainers/progressive_trainer.py --test --which_iter 2571482  --N 10 --dataset ade_indoor \
--nums_fid 433 --name ade_indoor_im --crop_size 256
