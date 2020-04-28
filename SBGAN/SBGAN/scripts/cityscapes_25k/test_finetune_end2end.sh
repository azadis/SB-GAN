#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN+SPADE model
# By Samaneh Azadi
# ============================================================

#cityscapes_25k

gpu_ids=0
name=ft_cityscapes_25k_end2end
dataset=cityscapes_full_weighted
dataroot=datasets
ckpt=weights
spade_epoch=61
pro_iter=540857
D2_iter=545959
bs=1
load_size=512
crp_size=512
num_semantics=20
label_seg=19
nums_fid=500

CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_end2end_trainer.py --test --name ${name} --dataset ${dataset}\
  --dataset_mode ${dataset} --dataroot ${dataroot} --num_semantics ${num_semantics} --label_seg ${label_seg} \
  --nums_fid ${nums_fid} --which_iter_D2 ${D2_iter} --which_iter ${pro_iter} --which_epoch ${spade_epoch} \
  --load_size ${load_size} --crop_size ${crp_size} --checkpoints_dir ${ckpt} --batchSize ${bs} --N 100 --no_instance --not_sort\
