#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================


#cityscapes_25k
CUDA_VISIBLE_DEVICES=1 python SBGAN/trainers/progressive_trainer.py --test --which_iter 1040199\
 --N 10 --dataset cityscapes_full --nums_fid 500 --name cityscapes_25k_im --max_dim 256 --crop_size 512 --aspect_ratio 2\
