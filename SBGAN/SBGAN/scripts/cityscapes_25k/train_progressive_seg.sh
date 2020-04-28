#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on segmentations
# By Samaneh Azadi
# ============================================================

#cityscapes_25k
gpu_ids=8,9
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_trainer.py \
 --name cityscapes_25k_segments --dataset cityscapes_full_weighted --save_freq 10000 --label_seg 34 --num_semantics 35\
 --epochs 20 --batchSize 32 
