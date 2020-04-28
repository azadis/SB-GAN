#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on RGB images directly
# By Samaneh Azadi
# ============================================================

#ADE_indoor
gpu_ids=0,1
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_trainer.py --name ade_indoor_im \
--dataset ade_indoor --save_freq 5000 --epochs 400 --batchSize 32

