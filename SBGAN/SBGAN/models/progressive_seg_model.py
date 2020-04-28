import datetime
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
                    
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from SBGAN.models.progressive import ProGANGenerator
from SBGAN.models.progressive import ProGANDiscriminator
from SBGAN.modules.gan_loss import ImprovedWGANLoss
from SBGAN.models import print_network

import PIL
import numpy as np
import scipy.io as sio
from matplotlib.colors import ListedColormap


class ProgressiveSegModel(torch.nn.Module):
    def __init__(self, opt, n_samples, D_inputs=0, load_all=True):
        super().__init__()

        self.opt = opt
        self.global_iteration = 0
        self.phase = "fade"
        self.batch_size = [opt.batchSize]*(int(np.log2(opt.max_dim)))
        if not D_inputs:
            D_inputs = self.opt.num_semantics
        
        if self.opt.train:
            for d in range(2, int(np.log2(opt.max_dim))+1):
                if 2**(d) > 128:
                    n_gpus = int(torch.cuda.device_count())
                    if self.batch_size[d-2]/n_gpus>=4 and int(self.batch_size[d-2]/n_gpus)%4==0:
                        self.batch_size[d-2]/=4

                    elif self.batch_size[d-2]/n_gpus>=3 and int(self.batch_size[d-2]/n_gpus)%3==0:
                        self.batch_size[d-2]/=3

                    elif self.batch_size[d-2]/n_gpus>=2 and int(self.batch_size[d-2]/n_gpus)%2==0:
                        self.batch_size[d-2]/=2
                # if 2**(d) > 128 and self.batch_size[d-2]>=2:
                #     self.batch_size[d-2]/=2

                self.batch_size[d-2] = int(self.batch_size[d-2])

        self.steps_per_phase = [int(opt.epochs*n_samples/self.batch_size[d]) for d in range(int(np.log2(opt.max_dim)))]#50000
        step_multiplier = [1, 1, 1, 1, 1, 1, 1, 1]
        self.steps_per_phase = [step_multiplier[k]*self.steps_per_phase[k] for k in range(len(self.steps_per_phase)) ]


        self.colormap = self.create_colormap(opt)
        self.writer = SummaryWriter(
            "%s/logs/"%self.opt.save_path + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        )
        self.generator, self.discriminator  = self.initialize_networks(D_inputs)
        self.opt_g, self.opt_d = self.create_optimizers(opt)
        self.gan_loss = ImprovedWGANLoss(self.discriminator)
        self.r_itr, self.num_semantics, self.dim, self.phase = self.load_networks(n_samples, load_all=load_all)


    def forward(self,iteration, global_iteration, dim_ind, 
                interpolate=False, z=torch.Tensor([]),  im_mc=torch.Tensor([]), im=torch.Tensor([]),alpha=None, mode='',hard=True):
        
        z = z.cuda()
        im_mc = im_mc.cuda()
        im = im.cuda()

        if mode == 'generator':
            g_loss, x_fake_mc = self.compute_generator_loss(iteration, global_iteration, dim_ind, 
                                interpolate=False, z=z, hard=hard)
            return g_loss, x_fake_mc

        if mode == 'discriminator':
            d_loss, x_fake_mc = self.compute_discriminator_loss(im_mc, im, z ,iteration, global_iteration,dim_ind,
                    interpolate=False, hard=hard)
            return d_loss, x_fake_mc

        if mode == 'inference':
            x_fake, x_fake_mc = self.generate_fake(alpha, z, global_iteration, hard=hard)
            return x_fake, x_fake_mc


    def initialize_networks(self, D_inputs=0):
        if 'cityscapes' in self.opt.dataset:
            aspect_ratio = 2
        else:
            aspect_ratio = 1
        if not D_inputs:
            D_inputs = self.opt.num_semantics

        generator = ProGANGenerator(max_dim=self.opt.max_dim, rgb=self.opt.rgb, num_semantics=self.opt.num_semantics, T=self.opt.T, aspect_ratio=aspect_ratio).cuda()
        discriminator = ProGANDiscriminator(max_dim=self.opt.max_dim, rgb=self.opt.rgb, num_semantics=D_inputs, aspect_ratio=aspect_ratio).cuda()
        print_network(generator)
        print_network(discriminator)
        return generator, discriminator


    def load_networks(self, n_samples, load_all=True):
        r_itr = 0
        dim = 4
        num_semantics_ = self.opt.num_semantics
        phase = "fade"
        self.generator.res = dim
        if self.opt.cont_train or not self.opt.train:
            self.global_iteration = self.opt.which_iter
            dim, phase = self.load_model(self.global_iteration, load_all=load_all)
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
        return r_itr, num_semantics_, dim, phase


    def create_optimizers(self,opt):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=opt.lr_pgan, betas=(0.0, 0.99)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt.lr_pgan, betas=(0.0, 0.99)
        )
        return opt_g, opt_d


    def create_colormap(self, opt):
        if 'indoor' in opt.dataset:
            colormat = sio.loadmat('datasets/color_indoor.mat')
            colormap = colormat['colors']

        elif 'cityscapes' in opt.dataset:
            colormat = sio.loadmat('datasets/cityscapes_color%s.mat'%opt.num_semantics)
            colormap = colormat['colors']
        return colormap


    def compute_generator_loss(self, iteration, global_iteration, dim_ind, 
                                interpolate=False, z=torch.Tensor([]), hard=True):
        if z.nelement() == 0:
            z = torch.randn(self.batch_size[dim_ind], 512).cuda()
        if interpolate:
            alpha = iteration / self.steps_per_phase[dim_ind]
            fake = self.generator.interpolate(z, alpha)
            # fake = sum_subset(fake, num_semantics)
            _, x_fake_mc = self.gumbelSampler(fake)
            score = self.discriminator.interpolate(x_fake_mc, alpha)
        else:
            alpha = 0
            fake = self.generator(z)
            # fake = sum_subset(fake, self.opt.num_semantics)
            _, x_fake_mc = self.gumbelSampler(fake)
            score = self.discriminator(x_fake_mc)
        logprob = torch.log(fake + 0.00001)
        entropy = -torch.mean(torch.sum(torch.mul(fake, logprob), dim=1))
        self.writer.add_scalar(
                            "avg_entropy",
                            torch.mean(entropy),
                            global_iteration,
                        )

        loss = self.gan_loss.generator_loss_logits(score)
        if hard:
            return loss, x_fake_mc
        else:
            return loss, fake


    def compute_discriminator_loss(self, im_mc, im, z ,iteration, global_iteration,dim_ind,
                         interpolate=False, hard=True):
        if (iteration + 1) % 10 == 0:

            im_ = self.color_transfer(im)
            grid = make_grid(im_[0:16,:,:,:], nrow=4, normalize=True, range=(-1, 1))
            self.writer.add_image("real", grid, global_iteration)

        if interpolate:
            alpha = iteration / self.steps_per_phase[dim_ind]
            real_score = self.discriminator.interpolate(im_mc, alpha)
            with torch.no_grad():
                fake = self.generator.interpolate(z, alpha)
                _, x_fake_mc = self.gumbelSampler(fake)
                x_fake_mc.requires_grad = False

            fake_score = self.discriminator.interpolate(x_fake_mc.detach(), alpha)
            forward = lambda x: self.discriminator.interpolate(x, alpha)
        else:
            real_score = self.discriminator(im_mc)
            with torch.no_grad():
                fake = self.generator(z)
                _, x_fake_mc = self.gumbelSampler(fake)

                x_fake_mc.requires_grad = False
            fake_score = self.discriminator(x_fake_mc.detach())
            forward = self.discriminator
        loss = self.gan_loss.discriminator_loss_logits(
            im_mc, x_fake_mc.detach(), real_score, fake_score, forward=forward
        )
        if hard:
            return loss, x_fake_mc
        else:
            return loss, fake

    def generate_fake(self, alpha, z, global_iteration, hard=True, vis=True):
        with torch.no_grad():
            if alpha is not None:
                fake = self.generator.interpolate(z, alpha)
            else:
                fake = self.generator(z)

            x_fake = fake.max(1, keepdim=True)[1]
            x_fake_mc = torch.zeros_like(fake).scatter_(1, x_fake, 1.0)

            x_fake[x_fake > self.opt.num_semantics-1]= 0
            x_fake = self.color_transfer(x_fake)

        fake = x_fake.cpu()
        if vis:
            grid = make_grid(fake, nrow=4, normalize=True, range=(-1, 1))
            self.writer.add_image("fake", grid, global_iteration)
        return x_fake, x_fake_mc


    def gumbelSampler(self, fake, hard=True, eps=1e-10, dim=1):
        
        logits = torch.log(fake+0.00001)
        if torch.isnan(logits.max()).data:
            print(fake.min(),fake.max())
        if eps != 1e-10:
                warnings.warn("`eps` parameter is deprecated and has no effect.")

        gumbels = -(torch.empty_like(logits).exponential_()+eps).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.opt.T  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = (y_hard - y_soft).detach() + y_soft
            return index.type(torch.cuda.FloatTensor), ret
        else:
            # Reparametrization trick.
            ret = y_soft
            return 0, ret


    def color_transfer(self, im):
        im = im.cpu().numpy()
        im_new = torch.Tensor(im.shape[0],3,im.shape[2], im.shape[3])
        newcmp = ListedColormap(self.colormap/255.0)
        for i in range(im.shape[0]):
            img = (im[i,0,:,:]).astype('uint8')
            # misc.imsave('/home/sazadi/bw.png', img)
            rgba_img = newcmp(img)
            rgb_img = PIL.Image.fromarray((255*np.delete(rgba_img, 3, 2)).astype('uint8'))
            tt = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            rgb_img = tt(rgb_img)
            # misc.imsave('/home/sazadi/seg.png',rgb_img.data.numpy().transpose(1,2,0))
            im_new[i,:,:,:] = rgb_img
        im_new = im_new.cuda()
        return im_new

    def save_model(self, num_semantics, global_iter, phase):
        # Save a model's weights, optimizer, and the state_dict
        if not os.path.isdir('%s/weights'%self.opt.save_path):
            os.makedirs('%s/weights'%self.opt.save_path)
        if not os.path.isdir('%s/weights/%s'%(self.opt.save_path, self.opt.name)):
            os.mkdir('%s/weights/%s'%(self.opt.save_path,self.opt.name))
        torch.save({
            'iter': global_iter,
            'phase': phase,
            'n_semantics': num_semantics,
            'G_state_dict': self.generator.state_dict(),
            'D_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'dim': self.generator.res}, '%s/weights/%s/%s.pth'%(self.opt.save_path, self.opt.name, global_iter))


    def load_model(self, global_iter, load_all=True):
        # load params
        checkpoint = torch.load('%s/weights/%s/%s.pth'%(self.opt.save_path, self.opt.name, global_iter))
        res = checkpoint['dim']
        # num_semantics = checkpoint['n_semantics']
        phase = checkpoint['phase']
        if not load_all:
            selected_pretrained_D_dict = {}
            new_blocks = []
            self.min_scale_end2end = int(np.log2(self.opt.min_res_end2end) - 2)
            self.max_scale_end2end = int(np.log2(self.opt.max_res_end2end) - 2)
            self.max_scale = int(np.log2(self.opt.max_dim ) -2)
            for i in range(self.min_scale_end2end, self.max_scale_end2end+1):
                new_blocks += ['blocks.%s'%(self.max_scale - i)] 
            pretrained_D_dict = {k: v for k, v in checkpoint['D_state_dict'].items()}
            for k,v in pretrained_D_dict.items():
                include_k = True                
                for id_ in new_blocks:
                    if id_ in k:
                        include_k = False
                if include_k:
                    selected_pretrained_D_dict[k] = v
        else:
            selected_pretrained_D_dict = checkpoint['D_state_dict']
        print(selected_pretrained_D_dict.keys())


        self.generator.load_state_dict((checkpoint['G_state_dict']))
        self.discriminator.load_state_dict(selected_pretrained_D_dict, strict=False)

        self.generator.res = res
        self.discriminator.res = res

        global_iter = checkpoint['iter']
        if not self.opt.train:
            self.generator.eval()
            self.discriminator.eval()
        return res, phase



