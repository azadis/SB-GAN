#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================

#cityscapes
CUDA_VISIBLE_DEVICES=1 python SBGAN/trainers/progressive_trainer.py --test --which_iter 173395\
--N 100 --dataset cityscapes --nums_fid 500 --name cityscapes_im --crop_size 512 --aspect_ratio 2
