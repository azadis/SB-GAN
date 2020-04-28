import datetime
                    
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, main_dir) 

from SBGAN.models.progressive import ProGANGenerator
from SBGAN.models.progressive import ProGANDiscriminator
from SBGAN.modules.gan_loss import ImprovedWGANLoss
from SBGAN import data
from SBGAN.modules.parameter import parameters
from SBGAN.models import print_network
from SBGAN.models.progressive_model import ProgressiveModel
import PIL
import numpy as np
import torch.nn as nn
from SBGAN.data.custom_dataset import RandomCropLongEdge
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
from scipy import misc

import scipy.io as sio
import time
from SPADE.models.networks.sync_batchnorm import DataParallelWithCallback
from SBGAN.modules.inception import InceptionV3
from SBGAN.modules.fid_score import calculate_fid_given_acts, get_activations
from torch.autograd import Variable


class ProgressiveTrainer:
    def __init__(self,opt):
        self.dataset = self.create_dataset(opt)
        n_samples = len(self.dataset)

        self.progressive_model = ProgressiveModel(opt, n_samples)
        self.opt = opt

        if len(opt.gpu_ids) > 0:
            self.progressive_model = DataParallelWithCallback(self.progressive_model,
                                                          device_ids=opt.gpu_ids)
            self.progressive_model_on_one_gpu = self.progressive_model.module
        else:
            self.progressive_model_on_one_gpu = self.progressive_model

    def create_dataset(self, opt):
        from SBGAN.data.custom_dataset_data_loader import CreateDataset

        main_data = opt.dataset
        opt.dataset= '%s_im'%main_data        
        dataset = CreateDataset(opt)
        return dataset


    def set_data_resolution(self, dim):
        ind = int(np.log2(dim))-2
        transform = []
        if 'cityscapes' not in self.opt.dataset:
            transform = [RandomCropLongEdge()]

        transform += [transforms.Resize(dim, PIL.Image.BILINEAR),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 

        transform = transforms.Compose(transform)

        self.dataset.transform = transform
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.progressive_model_on_one_gpu.batch_size[ind],
            shuffle=self.opt.train,
            num_workers=4,
            drop_last=True)

        self.loader_iter = iter(self.loader)

    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            return next(self.loader_iter)

    def step_generator(self, dim_ind, phase, iteration, global_iteration):
        self.progressive_model_on_one_gpu.opt_g.zero_grad()
        z = torch.randn(self.progressive_model_on_one_gpu.batch_size[dim_ind], 512)
        loss_gen, fake = self.progressive_model(iteration, global_iteration, dim_ind,
                            z=z, interpolate=phase == "fade", mode='generator')
        
        loss_gen = loss_gen.sum()
        self.progressive_model_on_one_gpu.writer.add_scalar(
            "loss/gen",
            loss_gen,
            global_iteration,
        )
        loss_gen.backward()
        self.progressive_model_on_one_gpu.opt_g.step()

    def step_discriminator(self, dim_ind, phase, iteration, global_iteration):
        self.progressive_model_on_one_gpu.opt_d.zero_grad()
        z = torch.randn(self.progressive_model_on_one_gpu.batch_size[dim_ind], 512)
        im, _ = self.next_batch()
        loss_dis,_ = self.progressive_model(iteration, global_iteration, dim_ind, z=z, interpolate=phase == "fade",
                                            im=im.cpu(), mode='discriminator')
        loss_dis = loss_dis.sum()

        self.progressive_model_on_one_gpu.writer.add_scalar(
            "loss/dis",
            loss_dis,
            global_iteration,
        )
        loss_dis.backward()
        self.progressive_model_on_one_gpu.opt_d.step()


    def run(self):

        fixed_z = torch.randn(16, 512)
        dim_ind = int(np.log2(self.progressive_model_on_one_gpu.dim))-2
        global_iteration = self.progressive_model_on_one_gpu.global_iteration
        res_init = self.progressive_model_on_one_gpu.generator.res
        while self.progressive_model_on_one_gpu.generator.res <= self.opt.max_dim:
            self.set_data_resolution(self.progressive_model_on_one_gpu.generator.res)
            print(f"Training at resolution {self.progressive_model_on_one_gpu.generator.res} with phase {self.progressive_model_on_one_gpu.phase}")
            
            for phase in ["fade", "stabilize"]:
            # for phase in ["stabilize"]:
                if self.progressive_model_on_one_gpu.generator.res == 4 and phase == "fade":
                    continue
                if res_init == self.progressive_model_on_one_gpu.generator.res:
                    # i_start = self.progressive_model_on_one_gpu.r_itr
                    if self.progressive_model_on_one_gpu.phase == "stabilize" and phase =="fade":
                            continue
                # else:
                #     i_start = 0
                for iteration in np.arange(0, self.progressive_model_on_one_gpu.steps_per_phase[dim_ind]):

                    self.step_generator(dim_ind, phase, iteration, global_iteration)

                    self.step_discriminator(dim_ind, phase, iteration, global_iteration)
                    # print('disc', time.time()-t3)
                    global_iteration += 1
                    if (iteration + 1) % 10 == 0:
                        alpha = (
                            iteration / self.progressive_model_on_one_gpu.steps_per_phase[dim_ind]
                            if phase == "fade"
                            else None
                        )
                        fake = self.progressive_model(iteration, global_iteration, dim_ind, z=fixed_z, alpha=alpha, mode='inference')
                    if (iteration + 1) % 100 == 0:
                        print(
                            f"Res {self.progressive_model_on_one_gpu.generator.res:03d}, {phase.rjust(9)}: Iteration {iteration + 1:05d}/{self.progressive_model_on_one_gpu.steps_per_phase[dim_ind]:05d}"
                        )

                    if (iteration + 1) %self.opt.save_freq ==0 or\
                        (iteration+1) == self.progressive_model_on_one_gpu.steps_per_phase[dim_ind]:
                        self.progressive_model_on_one_gpu.save_model(global_iteration, phase)
            self.progressive_model_on_one_gpu.generator.res *= 2
            self.progressive_model_on_one_gpu.discriminator.res *= 2
            dim_ind += 1

    
    def test_model(self ,global_iteration, n_sample=1000, alpha=None):

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.cuda()
        self.set_data_resolution(int(self.opt.crop_size/self.opt.aspect_ratio))
        self.progressive_model_on_one_gpu.eval()

        print('generator res: %s'%self.progressive_model_on_one_gpu.generator.res)
        print('generator phase: %s'%self.progressive_model_on_one_gpu.phase)
        if not os.path.isdir('samples'):
            os.mkdir('samples')
        if not os.path.isdir('samples/%s'%self.opt.name):
            os.mkdir('samples/%s'%self.opt.name)

        num_bs = 1
        for i in range(int(n_sample/num_bs)):
            print(i)
            z = torch.randn(num_bs, 512).cuda()
            global_iteration = self.opt.which_iter
            iteration = 0

            dim_ind = int(np.log2(self.progressive_model_on_one_gpu.dim))-2
            fake = self.progressive_model(iteration, global_iteration, dim_ind, z=z, mode='inference')
            # print(fake.size())
            fake = fake.cpu()
            for j in range(num_bs):
                save_image(fake[j,:,:,:], 'samples/%s/%s_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model_on_one_gpu.dim, global_iteration),
                             nrow=1, normalize=True, range=(-1,1))


        fid = self.compute_FID(alpha)
        print('fid: %s', fid)

    def compute_FID(self, alpha=None):
        num_ims = 0
        dims = 2048
        batchsize = 1 
        all_reals = np.zeros((int(self.opt.nums_fid/batchsize)*batchsize,dims))
        all_fakes = np.zeros((int(self.opt.nums_fid/batchsize)*batchsize,dims))
        print('fid at res %s'%int(self.opt.crop_size/self.opt.aspect_ratio))
        scaling = int(self.opt.crop_size / (self.opt.aspect_ratio * self.progressive_model_on_one_gpu.dim))
        upsample = nn.Upsample(scale_factor=scaling, mode='bilinear')

        for i in range(int(self.opt.nums_fid/batchsize)):

            real_ims, _ = self.next_batch()
            real_ims = Variable(real_ims).cuda()
            # real_ims = real_ims[:,:,::2,::2]
            real_acts = get_activations(real_ims, self.inception_model, batchsize, cuda=True)
            all_reals[i*batchsize:i*batchsize+real_acts.shape[0],:] = real_acts


            z = torch.randn(batchsize, 512).cuda()
            with torch.no_grad():
                if alpha is not None:
                    fake = self.progressive_model_on_one_gpu.generator.interpolate(z, alpha)
                else:
                    fake = self.progressive_model_on_one_gpu.generator(z)
            # fake = fake[:,:,::2,::2]
            fake = upsample(fake)
            fake_acts = get_activations(fake, self.inception_model, batchsize, cuda=True)
            all_fakes[i*batchsize:i*batchsize+fake_acts.shape[0],:] = fake_acts

            
        fid_eval = calculate_fid_given_acts(all_reals,all_fakes)
        return fid_eval



if __name__ == "__main__":
    opt = parameters().parse()

    opt.gpu_ids = list(range(int(torch.cuda.device_count())))

    opt.rgb = opt.rgb_seg

    if opt.test:
        opt.isTrain = False
        opt.train = False

        ProgressiveTrainer(opt).test_model(opt.which_iter, n_sample=opt.N)
    else:
        ProgressiveTrainer(opt).run()
