#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN+SPADE model
# By Samaneh Azadi
# ============================================================


#ade_indoor
gpu_ids=0
name=ft_ade_indoor_end2end
dataset=ade_indoor
dataroot=datasets/ade_indoor
ckpt=weights
spade_epoch=381
pro_iter=155851
D2_iter=156124
bs=1
load_size=256
crp_size=256
num_semantics=95
label_seg=95
nums_fid=433

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_end2end_trainer.py --test --name ${name} --dataset ${dataset}\
   --dataset_mode ${dataset} --dataroot ${dataroot} --num_semantics ${num_semantics} --label_seg ${label_seg} \
  --nums_fid ${nums_fid} --which_iter_D2 ${D2_iter} --which_iter ${pro_iter} --which_epoch ${spade_epoch} \
  --load_size ${load_size} --crop_size ${crp_size} --checkpoints_dir ${ckpt} --batchSize ${bs} --N 100 --no_instance\

