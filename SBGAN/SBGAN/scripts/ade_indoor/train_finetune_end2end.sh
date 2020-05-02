#!/bin/bash

# ============================================================
# Semantic Bottleneck GAN
# Script for fine-tuning the progressive GAN+SPADE model end2end
# By Samaneh Azadi
# ============================================================


#ade_indoor

gpu_ids=0,1,2,3,4,5,6,7
dataset=ade_indoor
name_joint=$dataset
name=ft_${name_joint}_end2end
dataroot=datasets/ade_indoor
ckpt=weights
spade_epoch=300
pro_iter=133738
D2_iter=0
lambda=10
bs=16
load_size=256
crp_size=256
num_semantics=95
label_seg=95
lr=0.0001
niter=50
niterd=50
nums_fid=433
save_freq=10
eval_freq=5

# =======================================
##COPY pretrained networks from their corresponding directories
# =======================================
if [ ! -d "${ckpt}/${name}" ]; then
    mkdir "${ckpt}/${name}"
fi

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
  --tf_log --no_instance --end2end --pretrain_D2 --continue_train --cont_train


