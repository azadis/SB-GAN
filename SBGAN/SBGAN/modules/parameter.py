import argparse

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, main_dir) 

from SPADE.options.train_options import TrainOptions


def str2bool(v):
    return v.lower() in ('true')

class parameters(TrainOptions):
    def initialize(self, parser):
    # parser = argparse.ArgumentParser()
        TrainOptions.initialize(self, parser)
        # Model hyper-parameters

        parser.add_argument('--train', type=str2bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--dataset', type=str, default='ADE_seg')#, choices=['cityscapes','cityscapes_full','ADE2s','ADE_indoor','ADE_sorted','ADE','ADE_seg','ADE_im','ADE_lbl','STL10'])
        parser.add_argument('--max_dim', type=int, default=256)
        parser.add_argument('--which_iter', type=int, default=1000, help='load the model in which iteration')
        parser.add_argument('--which_iter_D2', type=int, default=1000, help='load the D2 discriminator in which iteration')
        parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
        parser.add_argument('--N', type=int, default=1000, help='number of samples to generate')
        parser.add_argument('--nums_fid', type=int, default=1000, help='number of samples to generate to compute fid score')
        parser.add_argument('--min_res_end2end', type=int, default=4, help='number of samples to generate')
        parser.add_argument('--max_res_end2end', type=int, default=256, help='number of samples to generate')
        parser.add_argument('--eval_freq', type=int, default=10, help='frequency for evaluting fid')
        parser.add_argument('--test', action='store_true',help='run the model in eval phase and generate samples')
        parser.add_argument('--no_update_spade', action='store_true',help='do not train spade in the end to end model')
        parser.add_argument('--end2end', action='store_true',help='end2end with spade, it can run in two modes: either with/without no_update_spade')
        parser.add_argument('--save_path', type=str, default='./')
        parser.add_argument('--sample_path', type=str, default='fake_samples.npy')
        parser.add_argument('--sample_index', type=int, default=0)


        #paramteres with a different name in the SPADE params
        parser.add_argument('--rgb_seg', action='store_true',help='generate rgb segmentations')
        parser.add_argument('--label_seg', type=int, default=100, help='number of semantic labels in segmentation')
        parser.add_argument('--num_semantics', type=int, default=3, help='starting point for the number of semantic classes')
        parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving the model')
        parser.add_argument('--cont_train', action='store_true',help='continue training from global_iteration=iter')
        parser.add_argument('--T', type=float, default=1, help='Temperature in softmax')
        parser.add_argument('--lr_pgan', type=float, default=0.001, help='adam learning rate')
        parser.add_argument('--lambda_pgan', type=float, default=0.001, help='multiplier for the first network GAN loss terms')
        parser.add_argument('--lambda_rgb', type=float, default=0.1, help='multiplier for the first network GAN loss terms')
        parser.add_argument('--lambda_D2', type=float, default=0.1, help='multiplier for the 2nd discriminator in end2end finetuning')
        parser.add_argument('--conditional', action='store_true',help='discriminator of spade is used for end2end training with input-output pairs')
        parser.add_argument('--no_augment', action='store_true',help='use fake segmentations to finetune spade')
        parser.add_argument('--BN_eval', action='store_true',help='set batchnorm layers to eval mode')
        parser.add_argument('--update_pix2pix', action='store_true',help='update the params of the pix2pix model')
        parser.add_argument('--update_progan', action='store_true',help='update the params of the progan model')
        parser.add_argument('--update_pix2pix_w_D2', action='store_true',help='update pix2pix while updating D2')
        parser.add_argument('--update_progan_w_D2', action='store_true',help='update progan with extra D2 loss')
        parser.add_argument('--pretrain_D2', action='store_true',help='pretrain D2 from scratch')
        parser.add_argument('--z_notdefined', action='store_true',help='initial tensor z not defined at test time')

        return parser