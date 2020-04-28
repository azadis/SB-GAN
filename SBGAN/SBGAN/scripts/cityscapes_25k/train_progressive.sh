#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================

# cityscapes_25k
gpu_ids=0,1
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_trainer.py --name cityscapes_25k_im\
 --dataset cityscapes_full --save_freq 10000 --epochs 40 --batchSize 32

