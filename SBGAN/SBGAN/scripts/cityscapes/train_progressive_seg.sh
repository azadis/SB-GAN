#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for training the progressive GAN model on segmentations
# By Samaneh Azadi
# ============================================================

#cityscapes
gpu_ids=0,1
CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_trainer.py \
 --name cityscapes_segments --dataset cityscapes --save_freq 10000 --label_seg 34 --num_semantics 35\
 --epochs 150 --batchSize 16
 #--which_iter 298900 --cont_train