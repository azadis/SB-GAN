import datetime
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
                    
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from SBGAN.models.progressive import ProGANGenerator
from SBGAN.models.progressive import ProGANDiscriminator
from SBGAN.modules.gan_loss import ImprovedWGANLoss
from SBGAN.models import print_network

import numpy as np


class ProgressiveModel(torch.nn.Module):
    def __init__(self, opt, n_samples):
        super().__init__()

        self.opt = opt
        self.global_iteration = 0
        self.phase = "fade"
        self.batch_size = [opt.batchSize]*(int(np.log2(opt.max_dim)))
        self.opt.rgb = True
        
        if self.opt.isTrain:
            for d in range(2, int(np.log2(opt.max_dim))+1):
                if 2**(d) > 64:
                    self.batch_size[d-2]/=2
                if 2**(d) > 128:
                    self.batch_size[d-2]/=2
                self.batch_size[d-2] = int(self.batch_size[d-2])
        self.steps_per_phase = [int(opt.epochs*n_samples/self.batch_size[d]) for d in range(int(np.log2(opt.max_dim)))]#50000
        step_multiplier = [1, 1, 1, 1, 2, 2, 3, 4]
        self.steps_per_phase = [step_multiplier[k]*self.steps_per_phase[k] for k in range(len(self.steps_per_phase)) ]

        self.writer = SummaryWriter(
            "%s/logs/"%self.opt.save_path + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        )
        self.generator, self.discriminator,  = self.initialize_networks()
        self.opt_g, self.opt_d = self.create_optimizers(opt)
        self.gan_loss = ImprovedWGANLoss(self.discriminator)
        self.r_itr, self.dim, self.phase = self.load_networks(n_samples)


    def forward(self,iteration, global_iteration, dim_ind, 
                interpolate=False, z=torch.Tensor([]), im=torch.Tensor([]),alpha=None, mode=''):
        
        z = z.cuda()
        im = im.cuda()

        if mode == 'generator':
            g_loss, fake = self.compute_generator_loss(iteration, global_iteration, dim_ind, 
                                interpolate=False, z=z)
            return g_loss, fake

        if mode == 'discriminator':
            d_loss, fake = self.compute_discriminator_loss(im, z ,iteration, global_iteration,dim_ind,
                    interpolate=False)
            return d_loss, fake

        if mode == 'inference':
            fake = self.generate_fake(alpha, z, global_iteration)
            return fake


    def initialize_networks(self):
        if 'cityscapes' in self.opt.dataset:
            aspect_ratio = 2
        else:
            aspect_ratio = 1
        generator = ProGANGenerator(max_dim=self.opt.max_dim, rgb=self.opt.rgb, aspect_ratio=aspect_ratio).cuda()
        discriminator = ProGANDiscriminator(max_dim=self.opt.max_dim, rgb=self.opt.rgb, aspect_ratio=aspect_ratio).cuda()
        print_network(generator)
        print_network(discriminator)
        return generator, discriminator


    def load_networks(self, n_samples):
        r_itr = 0
        dim = 4
        phase = "fade"
        self.generator.res = dim
        if self.opt.cont_train or not self.opt.train:
            self.global_iteration = self.opt.which_iter
            dim, phase = self.load_model(self.global_iteration)
            self.generator.res = dim
            
            dim_ind = int(np.log2(dim))-2
            g_itr = 0
            steps_per_res = [0]*len(self.steps_per_phase)

            for i,j in enumerate(self.steps_per_phase):
                if i==0:
                    steps_per_res[i] = j
                else: 
                    steps_per_res[i] = j*2

            global_iters = np.cumsum(steps_per_res)
            if global_iters[dim_ind] > self.global_iteration:
                r_itr = global_iters[dim_ind] - self.global_iteration

            if phase == "stabilize":
                r_itr -= int(self.steps_per_phase[dim_ind]) 
        return r_itr, dim, phase


    def create_optimizers(self,opt):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=opt.lr_pgan, betas=(0.0, 0.99)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt.lr_pgan, betas=(0.0, 0.99)
        )
        return opt_g, opt_d


    def compute_generator_loss(self, iteration, global_iteration, dim_ind, 
                                interpolate=False, z=torch.Tensor([])):
        if z.nelement() == 0:
            z = torch.randn(self.batch_size[dim_ind], 512).cuda()
        if interpolate:
            alpha = iteration / self.steps_per_phase[dim_ind]
            fake = self.generator.interpolate(z, alpha)
            score = self.discriminator.interpolate(fake, alpha)
        else:
            fake = self.generator(z)
            score = self.discriminator(fake)
        loss = self.gan_loss.generator_loss_logits(score)

        return loss, fake


    def compute_discriminator_loss(self, im, z ,iteration, global_iteration,dim_ind,
                         interpolate=False):


        if (iteration + 1) % 10 == 0:
            grid = make_grid(im[0:16,:,:,:], nrow=4, normalize=True, range=(-1, 1))
            self.writer.add_image("real", grid, global_iteration)

        if interpolate:
            alpha = iteration / self.steps_per_phase[dim_ind]
            real_score = self.discriminator.interpolate(im, alpha)
            with torch.no_grad():
                fake = self.generator.interpolate(z, alpha)

            fake_score = self.discriminator.interpolate(fake.detach(), alpha)
            forward = lambda x: self.discriminator.interpolate(x, alpha)
        else:
            real_score = self.discriminator(im)
            with torch.no_grad():
                fake = self.generator(z)
            fake_score = self.discriminator(fake.detach())
            forward = self.discriminator
        loss = self.gan_loss.discriminator_loss_logits(
            im, fake.detach(), real_score, fake_score, forward=forward
        )
        return loss, fake

    def generate_fake(self, alpha, z, global_iteration):
        with torch.no_grad():
            if alpha is not None:
                fake = self.generator.interpolate(z, alpha)
            else:
                fake = self.generator(z)

        grid = make_grid(fake.cpu(), nrow=4, normalize=True, range=(-1, 1))
        self.writer.add_image("fake", grid, global_iteration)
        return fake


    def save_model(self, global_iter, phase):
        # Save a model's weights, optimizer, and the state_dict
        if not os.path.isdir('%s/weights'%self.opt.save_path):
            os.makedirs('%s/weights'%self.opt.save_path)
        if not os.path.isdir('%s/weights/%s'%(self.opt.save_path, self.opt.name)):
            os.mkdir('%s/weights/%s'%(self.opt.save_path,self.opt.name))
        torch.save({
            'iter': global_iter,
            'phase': phase,
            'G_state_dict': self.generator.state_dict(),
            'D_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'dim': self.generator.res}, '%s/weights/%s/%s.pth'%(self.opt.save_path, self.opt.name, global_iter))


    def load_model(self, global_iter):
        # load params
        checkpoint = torch.load('%s/weights/%s/%s.pth'%(self.opt.save_path, self.opt.name, global_iter))
        res = checkpoint['dim']
        phase = checkpoint['phase']

        self.generator.load_state_dict((checkpoint['G_state_dict']))
        self.discriminator.load_state_dict((checkpoint['D_state_dict']))
        self.opt_g.load_state_dict((checkpoint['opt_g_state_dict']))
        self.opt_d.load_state_dict((checkpoint['opt_d_state_dict']))

        self.generator.res = res
        self.discriminator.res = res

        global_iter = checkpoint['iter']
        if not self.opt.train:
            self.generator.eval()
            self.discriminator.eval()
        return res, phase



