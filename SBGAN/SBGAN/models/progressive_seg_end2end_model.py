import datetime
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, main_dir) 
                    
import torch
from torchvision import transforms
import torch.nn as nn

from SBGAN.models import print_network
from SBGAN.models.progressive_seg_model import ProgressiveSegModel

from SPADE.models.pix2pix_model import Pix2PixModel
import SPADE.models.networks as networks
import SPADE.util.util as util

import numpy as np

class ProgressiveSegEnd2EndModel(torch.nn.Module):
    def __init__(self, opt, n_samples):
        super().__init__()

        self.opt = opt
        self.max_scale = int(np.log2(self.opt.max_dim))-1
        D_inputs = [self.opt.num_semantics]*self.max_scale
        self.min_scale_end2end = int(np.log2(self.opt.min_res_end2end) - 2)
        self.max_scale_end2end = int(np.log2(self.opt.max_res_end2end) - 2)

        self.progressive_model = ProgressiveSegModel(opt, n_samples, D_inputs=D_inputs, load_all=True)
        self.pix2pix_model = Pix2PixModel(opt, end2end=opt.end2end)
        self.netD2 = self.initialize_networks()

    def initialize_networks(self):
        netD = networks.define_D2(self.opt) if self.opt.isTrain else None
        if self.opt.isTrain and self.opt.which_iter_D2>0:
            print_network(netD)
            netD = util.load_network(netD, 'D2', self.opt.which_iter_D2 , self.opt)

        return netD

    def create_optimizers(self, lr=None):
        if lr is None:
            lr = self.opt.lr
        opt = self.opt
        if self.opt.isTrain:
            D2_params = list(self.netD2.parameters())
            beta1, beta2 = 0, 0.99

            optimizer_D2 = torch.optim.Adam(D2_params, lr=lr, betas=(beta1, beta2))

            return optimizer_D2
        else:
            return None


    def forward(self,iteration, global_iteration, dim_ind, z,  im_mc, im_seg, im , scaling,
                interpolate=False, alpha=None, mode=''):
        z = z.cuda()
        im_mc = im_mc.cuda()
        im = im.cuda()
        im_seg = im_seg.cuda()

        if mode == 'generator_end2end':
            g_loss, fake_semantics = self.compute_end2end_generator_loss( im_mc, im, z, iteration, global_iteration, dim_ind, 
                                scaling, interpolate, hard=True)
            
            return g_loss, fake_semantics
        if mode == 'discriminator_end2end':
            d_loss = self.compute_end2end_discriminator_loss(im_mc, im_seg, im, z ,iteration, global_iteration,dim_ind,
                    scaling, interpolate,alpha, hard=True)
            return d_loss
        
        if mode == 'discriminator':
            d_loss = self.compute_discriminator2_loss(im_mc, im, z, global_iteration,
                        scaling, interpolate)
            return d_loss

        if mode == 'inference':
            fake_seg, fake_im_f, fake_im_r = self.log_images(im_mc, im, z, global_iteration, scaling, alpha)
            return fake_seg, fake_im_f, fake_im_r


    def discriminate(self, fake_image, real_image):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image ,real_image], dim=0)
        discriminator_out = self.netD2(fake_and_real)

        pred_fake, pred_real = self.pix2pix_model.divide_pred(discriminator_out)

        return pred_fake, pred_real



    def compute_generator_loss(self, iteration, global_iteration, dim_ind, scaling,
                                interpolate=False, z=torch.Tensor([]), hard=True):
        if z.nelement() == 0:
            z = torch.randn(self.progressive_model.batch_size[dim_ind], 512).cuda()
        if interpolate:
            alpha = iteration / self.progressive_model.steps_per_phase[dim_ind]
            fake = self.progressive_model.generator.interpolate(z, alpha)
            _, x_fake_mc = self.progressive_model.gumbelSampler(fake)
            score = self.progressive_model.discriminator.interpolate(x_fake_mc, alpha)

        else:
            alpha = 0
            fake = self.progressive_model.generator(z)
            _, x_fake_mc = self.progressive_model.gumbelSampler(fake)

            score = self.progressive_model.discriminator(x_fake_mc)

        logprob = torch.log(fake + 0.00001)
        entropy = -torch.mean(torch.sum(torch.mul(fake, logprob), dim=1))
        self.progressive_model.writer.add_scalar(
                            "avg_entropy",
                            torch.mean(entropy),
                            global_iteration,
                        )

        loss = self.progressive_model.gan_loss.generator_loss_logits(score)
        if hard:
            return loss, x_fake_mc
        else:
            return loss, fake




    def compute_end2end_generator_loss(self, real_semantics, real_image, z, iteration, global_iteration, dim_ind, 
                                scaling, interpolate=False, hard=True):
        G_losses = {}

        #fake seg
        g_loss_fake, fake_semantics = self.compute_generator_loss(iteration, global_iteration, dim_ind,scaling, 
                                interpolate, z=z, hard=hard)
        if self.opt.update_progan:
            G_losses['GAN_pro'] = g_loss_fake * self.opt.lambda_pgan

        
        #fff: fake from fake
        if self.opt.update_pix2pix_w_D2 or self.opt.update_progan_w_D2:
            upsample = nn.Upsample(scale_factor=scaling, mode='nearest')
            x_fake_mc_up = upsample(fake_semantics)

            fake_im_f, _ = self.pix2pix_model.generate_fake(x_fake_mc_up, real_image)
            pred_fake, pred_real = self.discriminate(fake_im_f, real_image)
            G_losses['GAN_fff'] = self.opt.lambda_D2*self.pix2pix_model.criterionGAN(pred_fake, True,
                                                for_discriminator=False)


        #ffr: fake from real
        if self.opt.update_pix2pix:
            g_loss, fake_im_r = self.pix2pix_model.compute_generator_loss(
                real_semantics, real_image)
            G_losses['GAN_ffr'] = g_loss['GAN'] 
            if not self.opt.no_ganFeat_loss:
                G_losses['GAN_Feat'] = g_loss['GAN_Feat']
            if not self.opt.no_vgg_loss:
                G_losses['VGG'] = g_loss['VGG']

        
        return G_losses, fake_semantics

    def compute_discriminator2_loss(self, real_semantics, real_image, z, global_iteration,
                    scaling, interpolate=False):

        D_losses = {}

        alpha = None
        x_fake, x_fake_mc = self.progressive_model.generate_fake(alpha, z, global_iteration, vis=False)
        upsample = nn.Upsample(scale_factor=scaling, mode='nearest')

        x_fake_mc = upsample(x_fake_mc)

        with torch.no_grad():
            fake_im_f, _ = self.pix2pix_model.generate_fake(x_fake_mc, real_image, compute_kld_loss=False)
            fake_im_f = fake_im_f.detach()
            fake_im_f.requires_grad_()
            x_fake_mc = x_fake_mc.detach()
            x_fake_mc.requires_grad_()

            fake_im_r, _ = self.pix2pix_model.generate_fake(
                real_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        pred_fake, pred_real = self.discriminate(fake_im_f, real_image)
        D_losses['D_Fake_fff'] = self.pix2pix_model.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.pix2pix_model.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses


    def compute_discriminator_loss(self, im_mc, im, z ,iteration, global_iteration,dim_ind,scaling,
                         interpolate=False, hard=True):
        # im_mc = nn.functional.interpolate(im_mc, size=[int(im_mc.size(2)/scaling), int(im_mc.size(3)/scaling)],
        #     mode='bilinear')

        im_mc = im_mc[:,:,::int(scaling),::int(scaling)]
        if interpolate:
            alpha = iteration / self.progressive_model.steps_per_phase[dim_ind]
            real_score = self.progressive_model.discriminator.interpolate(im_mc, alpha)
            with torch.no_grad():
                fake = self.progressive_model.generator.interpolate(z, alpha)
                _, x_fake_mc = self.progressive_model.gumbelSampler(fake)
                x_fake_mc.requires_grad = False

            fake_score = self.progressive_model.discriminator.interpolate(x_fake_mc.detach(), alpha)
            forward = lambda x: self.progressive_model.discriminator.interpolate(x, alpha)
            loss = self.progressive_model.gan_loss.discriminator_loss_logits(
                im_mc, x_fake_mc.detach(), real_score, fake_score, forward=forward
            )
        else:
            real_score = self.progressive_model.discriminator(im_mc)
            with torch.no_grad():
                fake = self.progressive_model.generator(z)
                _, x_fake_mc = self.progressive_model.gumbelSampler(fake)

                x_fake_mc.requires_grad = False
            fake_score = self.progressive_model.discriminator(x_fake_mc.detach())
            forward = self.progressive_model.discriminator
            loss = self.progressive_model.gan_loss.discriminator_loss_logits(
                im_mc, x_fake_mc.detach(), real_score, fake_score, forward=forward
            )
        if hard:
            return loss, x_fake_mc
        else:
            return loss, fake



    def compute_end2end_discriminator_loss(self, real_semantics, real_semantics_one, real_image, z ,iteration, global_iteration,dim_ind,
                    scaling, interpolate=False,alpha=None,  hard=True):
        D_losses = {}


        d_loss, x_fake_mc = self.compute_discriminator_loss(real_semantics, real_image, z ,iteration, global_iteration,dim_ind,
                    scaling, interpolate, hard=hard)
        if self.opt.update_progan:
            D_losses['GAN_pro'] = d_loss * self.opt.lambda_pgan


        if self.opt.update_pix2pix:
            d_loss = self.pix2pix_model.compute_discriminator_loss(
                        real_semantics, real_image)
            D_losses['D_Fake_ffr'] = d_loss['D_Fake']
            D_losses['D_real'] = d_loss['D_real']


        if self.opt.update_pix2pix_w_D2 or self.opt.update_progan_w_D2:
            upsample = nn.Upsample(scale_factor=scaling, mode='nearest')

            x_fake_mc = upsample(x_fake_mc)


            with torch.no_grad():
                fake_im_f, _ = self.pix2pix_model.generate_fake(x_fake_mc, real_image, compute_kld_loss=False)
                fake_im_f = fake_im_f.detach()
                fake_im_f.requires_grad_()
                x_fake_mc = x_fake_mc.detach()
                x_fake_mc.requires_grad_()

            pred_fake, pred_real = self.discriminate(fake_im_f, real_image)
            D_losses['D_Fake_fff'] = self.opt.lambda_D2*self.pix2pix_model.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] += self.opt.lambda_D2*self.pix2pix_model.criterionGAN(pred_real, True,
                                                   for_discriminator=True)
        return D_losses
    

    def log_images(self, real_semantics, real_im, z,global_iteration,scaling, alpha=None):
        x_fake, x_fake_mc = self.progressive_model.generate_fake(alpha, z, global_iteration, vis=False)
        fake_im_r = torch.Tensor(x_fake.size(0), 1).cuda()
        fake_im_f = torch.Tensor(x_fake.size(0), 1).cuda()

        with torch.no_grad():
            fake_im_r, _ = self.pix2pix_model.generate_fake(real_semantics, real_im)
        upsample = nn.Upsample(scale_factor=scaling, mode='nearest')
        with torch.no_grad():
            # if not self.opt.update_progan and not self.opt.update_progan_w_D2:
            self.pix2pix_model.eval()
            x_fake_mc_up = upsample(x_fake_mc)
            fake_im_f, _ = self.pix2pix_model.generate_fake(x_fake_mc_up, real_im)

        return x_fake,fake_im_f, fake_im_r 
        








