#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for fine-tuning the progressive GAN+SPADE model end2end
# By Samaneh Azadi
# ============================================================

# cityscapes_25k
gpu_ids=0,1,2,3,4,5,6,7
name_joint=cityscapes_25k
name=ft_${name_joint}_end2end
dataset=cityscapes_full_weighted
dataroot=datasets
ckpt=weights
spade_epoch=30
pro_iter=382695
D2_iter=0
lambda=10
bs=8
load_size=512
crp_size=512
num_semantics=20
label_seg=19
lr=0.00001
niter=30
niterd=30
nums_fid=500
save_freq=5
eval_freq=1


if [ ! -d "${ckpt}/${name}" ]; then
    mkdir "${ckpt}/${name}"
fi

# =======================================
##COPY pretrained networks from their corresponding directories
# =======================================
pro_gan_pretrained="${ckpt}/${name_joint}_segment" 
if [ ! -f "${ckpt}/${name}/${pro_iter}.pth" ]; then
    cp "${pro_gan_pretrained}/${pro_iter}.pth" "${ckpt}/${name}/"
fi

spade_pretrained="../SPADE/${ckpt}/${name_joint}"
if [ ! -f "${ckpt}/${name}/${spade_epoch}_net_G.pth" ]; then
    cp "${spade_pretrained}/${spade_epoch}_net_G.pth" "${ckpt}/${name}/"
    cp "${spade_pretrained}/${spade_epoch}_net_D.pth" "${ckpt}/${name}/"
fi


CUDA_VISIBLE_DEVICES=$gpu_ids python SBGAN/trainers/progressive_seg_end2end_trainer.py --name ${name} --dataset ${dataset}\
  --dataset_mode ${dataset} --dataroot ${dataroot} --num_semantics ${num_semantics} --label_seg ${label_seg} --lr_pgan ${lr} \
  --load_size ${load_size} --crop_size ${crp_size} --checkpoints_dir ${ckpt} --nums_fid ${nums_fid} \
  --eval_freq ${eval_freq} --save_epoch_freq ${save_freq} --batchSize ${bs}  --lambda_pgan ${lambda}\
  --which_epoch ${spade_epoch} --which_iter ${pro_iter} --which_iter_D2 ${D2_iter} --niter ${niter} --niter_decay ${niterd}\
  --tf_log --no_instance --end2end --pretrain_D2 --continue_train --cont_train --not_sort 
