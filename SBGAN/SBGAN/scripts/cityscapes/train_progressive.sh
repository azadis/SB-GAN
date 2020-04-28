#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================

# cityscapes
gpu_ids=0,1
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_trainer.py --name cityscapes_im\
 --dataset cityscapes --save_freq 10000 --epochs 400 --batchSize 32 \
#--cont_train --which_iter 920199 
