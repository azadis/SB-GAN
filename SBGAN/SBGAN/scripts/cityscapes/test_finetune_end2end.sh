#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for testing the progressive GAN+SPADE model
# By Samaneh Azadi
# ============================================================


#cityscapes
gpu_ids=0
name=ft_ade_indoor_end2end
dataset=cityscapes
dataroot=datasets/cityscapes
ckpt=weights
spade_epoch=200
pro_iter=298900
D2_iter=325612
bs=1
load_size=512
crp_size=512
num_semantics=35
label_seg=35
nums_fid=500

CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_end2end_trainer.py --test --name ${name} --dataset ${dataset}\
   --dataset_mode ${dataset} --dataroot ${dataroot} --num_semantics ${num_semantics} --label_seg ${label_seg} \
  --nums_fid ${nums_fid} --which_iter_D2 ${D2_iter} --which_iter ${pro_iter} --which_epoch ${spade_epoch} \
  --load_size ${load_size} --crop_size ${crp_size} --checkpoints_dir ${ckpt} --batchSize ${bs} --N 1000 --no_instance\

