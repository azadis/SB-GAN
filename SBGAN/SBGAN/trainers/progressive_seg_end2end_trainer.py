import datetime
                    
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, main_dir) 

from SBGAN.modules.gan_loss import ImprovedWGANLoss
from SBGAN import data
from SBGAN.modules.parameter import parameters
from SBGAN.models import print_network
from SBGAN.models.progressive_seg_end2end_model import ProgressiveSegEnd2EndModel
from SBGAN.modules.inception import InceptionV3
from SBGAN.modules.fid_score import calculate_fid_given_acts, get_activations

from SPADE.models.networks.sync_batchnorm import DataParallelWithCallback
from SPADE.util.iter_counter import IterationCounter
import SPADE.util.util as util


import PIL
import numpy as np


class ProgressiveTrainer:
    def __init__(self,opt):
        self.opt = opt
        self.create_dataset()
        self.n_samples = len(self.dataset)

        self.end2end_model = ProgressiveSegEnd2EndModel(opt, self.n_samples)

        #end2end model
        if len(opt.gpu_ids) > 0:
            self.end2end_model = DataParallelWithCallback(self.end2end_model,
                                                          device_ids=opt.gpu_ids)
            self.end2end_model_on_one_gpu = self.end2end_model.module
        else:
            self.end2end_model_on_one_gpu = self.end2end_model
        self.progressive_model = self.end2end_model_on_one_gpu.progressive_model
        self.pix2pix_model = self.end2end_model_on_one_gpu.pix2pix_model 

        self.pix2pix_model.optimizer_G, self.pix2pix_model.optimizer_D = \
            self.pix2pix_model.create_optimizers(opt)
        self.optimizer_D2 = self.end2end_model_on_one_gpu.create_optimizers()
        self.phase = "stabilize"

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.cuda()


    def create_dataset(self):
        from SBGAN.data.custom_dataset_data_loader import CreateDataset
        self.dataset = CreateDataset(self.opt, train=True)
        self.dataset_eval = CreateDataset(self.opt, train=False)


    def set_data_resolution(self, dim):


        transform = [transforms.Resize(dim, PIL.Image.NEAREST)]
        transform += [transforms.RandomHorizontalFlip(p=1)]
        transform += [transforms.ToTensor()] 


        transform_im = [transforms.Resize(dim, PIL.Image.BILINEAR)]
        transform_im += [transforms.RandomHorizontalFlip(p=1)]
        transform_im += [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 

        self.dataset.transform_lbl = transform
        self.dataset.transform_im = transform_im
        self.dataset_eval.transform_lbl = transform
        self.dataset_eval.transform_im = transform_im

        opt = self.opt
        self.loader_data = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt.batchSize,
            shuffle=True,
            num_workers=4,
            drop_last=True)

        self.loader_data_eval = torch.utils.data.DataLoader(
            self.dataset_eval,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=True)

        self.loader_iter = iter(self.loader_data)
        self.loader_iter_eval = iter(self.loader_data_eval)

    def next_batch(self):
        try:
            X_seg, X_im, Y = next(self.loader_iter)
            return X_seg, Y, X_im, Y
        except StopIteration:
            self.loader_iter = iter(self.loader_data)
            X_seg, X_im, Y = next(self.loader_iter)
            return X_seg, Y, X_im, Y

    def next_batch_eval(self):
        try:
            X_seg, X_im, Y = next(self.loader_iter_eval)
            return X_seg, Y, X_im, Y
        except StopIteration:
            self.loader_iter_eval = iter(self.loader_data_eval)
            X_seg, X_im, Y = next(self.loader_iter_eval)
            return X_seg, Y, X_im, Y


    def call_next_batch(self, seg, im):
        seg= seg.cpu().numpy()
        if 'cityscapes' in self.opt.dataset and self.num_semantics==19:
            seg += 1
            seg = np.where(seg>self.num_semantics-1, 0, seg)
        seg = torch.Tensor(seg).cuda()
        seg_mc = torch.FloatTensor(self.opt.batchSize, self.num_semantics, seg.size(2), seg.size(3)).zero_().cuda()
        seg_mc = seg_mc.scatter_(1,seg.long(),1.0)
        return seg, seg_mc, im 

    def step_discriminator(self, iteration, global_iteration, dim_ind,seg_mc,seg, im, scaling, phase):
        self.optimizer_D2.zero_grad()

        z = torch.randn(self.opt.batchSize, 512)
        D_losses = self.end2end_model(iteration, global_iteration,dim_ind, z, seg_mc.cpu(),seg.cpu(), im.cpu(),
                        scaling, interpolate=phase == "fade", mode='discriminator')

        D_losses = sum(D_losses.values()).mean()
        self.progressive_model.writer.add_scalar(
            "loss/dis_D2",
            D_losses,
            global_iteration,
        )

        D_losses.backward()

        self.optimizer_D2.step()
        return D_losses


    def step_generator_end2end(self, iteration, global_iteration, dim_ind, seg_mc, seg, im, scaling, phase):
        self.progressive_model.opt_g.zero_grad()
        self.pix2pix_model.optimizer_G.zero_grad()


        z = torch.randn(self.opt.batchSize, 512)
        G_losses, fake_seg = self.end2end_model(iteration, global_iteration, dim_ind, z, seg_mc.cpu(), seg.cpu(), im.cpu(), 
                scaling, interpolate=phase == "fade", mode='generator_end2end')
        
        if self.opt.update_progan:
            self.progressive_model.writer.add_scalar(
                "loss/G_proGAN",
                G_losses['GAN_pro'].mean(),global_iteration,
            )

        self.progressive_model.writer.add_scalar(
                "loss/gen",
                sum(G_losses.values()).mean(), global_iteration,
            )

        if self.opt.update_pix2pix:
            self.progressive_model.writer.add_scalar(
                    "loss/GAN_ffr",
                    (G_losses['GAN_ffr']).mean(),global_iteration,
                )
            self.progressive_model.writer.add_scalar(
                    "loss/GAN_Feat",
                    (G_losses['GAN_Feat']).mean(),global_iteration,
                )

            self.progressive_model.writer.add_scalar(
                    "loss/VGG",
                    (G_losses['VGG']).mean(),global_iteration,
                )

        if self.opt.update_pix2pix_w_D2 or self.opt.update_progan_w_D2:
            self.progressive_model.writer.add_scalar(
                    "loss/GAN_fff",
                    (G_losses['GAN_fff']).mean(),global_iteration,
                )


        G_losses = sum(G_losses.values()).mean()
        G_losses.backward()

        if self.opt.update_progan or self.opt.update_progan_w_D2:
            self.progressive_model.opt_g.step()
        if self.opt.update_pix2pix or self.opt.update_pix2pix_w_D2:
            self.pix2pix_model.optimizer_G.step()

        return G_losses

    def step_discriminator_end2end(self, iteration, global_iteration, dim_ind, seg_mc, seg, im, scaling, phase):

        self.progressive_model.opt_d.zero_grad()
        self.pix2pix_model.optimizer_D.zero_grad()
        self.optimizer_D2.zero_grad()

        z = torch.randn(self.opt.batchSize, 512)
        D_losses = self.end2end_model(iteration, global_iteration,dim_ind, z, seg_mc.cpu(), seg.cpu(), im.cpu(),
                        scaling, interpolate=phase == "fade", mode='discriminator_end2end')

        if self.opt.update_progan:
            self.progressive_model.writer.add_scalar(
                    "loss/D_proGAN",
                    (D_losses['GAN_pro']).mean(),global_iteration,
                )

 
        self.progressive_model.writer.add_scalar(
                "loss/dis",
                sum(D_losses.values()).mean(),global_iteration,
            )

        if self.opt.update_progan_w_D2 or self.opt.update_pix2pix_w_D2:
            self.progressive_model.writer.add_scalar(
                    "loss/D_Fake_fff",
                    (D_losses['D_Fake_fff']).mean(),global_iteration,
                )

        if self.opt.update_pix2pix:
            self.progressive_model.writer.add_scalar(
                    "loss/D_Fake_ffr",
                    (D_losses['D_Fake_ffr']).mean(),global_iteration,
                )

        if self.opt.update_progan_w_D2 or self.opt.update_pix2pix_w_D2 or self.opt.update_pix2pix:
            self.progressive_model.writer.add_scalar(
                    "loss/D_real",
                    (D_losses['D_real']).mean(),global_iteration,
                )

        D_losses = sum(D_losses.values()).mean()
        D_losses.backward()
        if self.opt.update_progan:
            self.progressive_model.opt_d.step()
        if self.opt.update_pix2pix_w_D2: # or self.opt.update_progan_w_D2:
            self.optimizer_D2.step()
        if self.opt.update_pix2pix:
            self.pix2pix_model.optimizer_D.step()

        return D_losses


    def run_pretrain(self, load=False):
        if not load:
            global_iteration = 0
            training_steps = int(self.n_samples /self.opt.batchSize)
            self.num_semantics = self.progressive_model.num_semantics
            self.set_data_resolution(int(self.opt.crop_size/self.opt.aspect_ratio))
            print(f"Training at resolution {self.progressive_model.generator.res}")
            dim_ind = 0
            phase = "stabilize"
            scaling = int(self.opt.crop_size / (self.opt.aspect_ratio * self.progressive_model.generator.res))
            num_epochs=1
            for epoch in range(num_epochs):
                for iteration in range(training_steps):
                    seg, _, im, _ = self.next_batch()
                    seg, seg_mc, im = self.call_next_batch(seg,im)
                    D_losses = self.step_discriminator( iteration, global_iteration, dim_ind, seg_mc, seg, im, scaling, phase)

                    global_iteration += 1
                    if (iteration + 1) % 10 == 0:
                        print(
                            f"Res {self.progressive_model.generator.res:03d}, {phase.rjust(9)}: Iteration {iteration + 1:05d}/{training_steps:05d}, epoch:{epoch + 1:05d}/{num_epochs:05d}"
                        )

                if epoch % self.opt.save_epoch_freq == 0 or \
                   (epoch+1) == num_epochs:
                    util.save_network(self.end2end_model_on_one_gpu.netD2, 'D2', global_iteration, self.opt)
        else:
            netD2 = self.end2end_model_on_one_gpu.netD2
            netD2 = util.load_network(netD2, 'D2', self.opt.which_iter_D2, self.opt)
            global_iteration = self.opt.which_iter_D2

        return global_iteration


    def run(self):

        iteration_D2 = self.run_pretrain(load=not self.opt.pretrain_D2)
        self.optimizer_D2 = self.end2end_model_on_one_gpu.create_optimizers(lr=self.opt.lr)

        fixed_z = torch.randn(self.opt.batchSize, 512)
        self.num_semantics = self.progressive_model.num_semantics
        global_iteration = self.progressive_model.global_iteration
        training_steps = int(self.n_samples /self.opt.batchSize)
        self.set_data_resolution(int(self.opt.crop_size/self.opt.aspect_ratio))
        print(f"Training at resolution {self.progressive_model.generator.res}")
        dim_ind = 0
        phase = "stabilize"
        scaling = int(self.opt.crop_size / (self.opt.aspect_ratio * self.progressive_model.generator.res))
        upsample = nn.Upsample(scale_factor=scaling, mode='nearest')
        z_fid = torch.randn(self.opt.nums_fid, 512).cuda()
        epoch_start = 0
        iter_counter = IterationCounter(self.opt, len(self.dataset))
        self.old_lr = self.opt.lr
        self.opt.epochs = self.opt.niter + self.opt.niter_decay

        if self.opt.BN_eval:
            for module in self.pix2pix_model.modules():
                if "BATCHNORM" in module.__class__.__name__.upper():
                    print(module.__class__.__name__)
                    module.eval()


        for epoch in range(self.opt.epochs):

            if epoch % self.opt.eval_freq == 0 or \
                epoch == self.opt.epochs:
                if not self.opt.BN_eval:
                    self.pix2pix_model.eval()
                fid = self.compute_FID(global_iteration, z_fixed=z_fid, real_fake='fake') #real_fake='real'/fake
                self.progressive_model.writer.add_scalar(
                        "fid_fake",
                        fid,global_iteration,
                    )
                fid = self.compute_FID(global_iteration, z_fixed=z_fid, real_fake='real') #real_fake='real'/fake
                self.progressive_model.writer.add_scalar(
                        "fid_real",
                        fid,global_iteration,
                    )

                if not self.opt.BN_eval:
                    self.pix2pix_model.train()

            iter_counter.record_epoch_start(epoch)
            for iteration in np.arange(training_steps):

                iter_counter.record_one_iteration()
                seg, _, im, _ = self.next_batch()
                seg, seg_mc, im = self.call_next_batch(seg,im)

                G_losses = self.step_generator_end2end(iteration, global_iteration, dim_ind, seg_mc, seg, im, scaling, phase)

                D_losses = self.step_discriminator_end2end(iteration, global_iteration, dim_ind, seg_mc, seg, im, scaling, phase)
                # print('disc', time.time()-t3)
                global_iteration += 1
                if (iteration + 1) % 100 == 0:
                    alpha = (
                        iteration / self.progressive_model.steps_per_phase[dim_ind]
                        if phase == "fade"
                        else None
                    )
                    fake_seg, fake_im_f, fake_im_r = self.end2end_model(iteration, global_iteration, dim_ind, fixed_z, seg_mc.cpu(), seg.cpu(), im.cpu(), scaling, mode='inference')
                    grid = make_grid(fake_seg, nrow=4, normalize=True, range=(-1, 1))
                    self.progressive_model.writer.add_image("fake", grid, global_iteration)

                    fake_im_f = fake_im_f.cpu()
                    grid = make_grid(fake_im_f, nrow=4, normalize=True, range=(-1, 1))
                    self.progressive_model.writer.add_image("fake_im_ff", grid, global_iteration)

                    fake_im_r = fake_im_r.cpu()
                    grid = make_grid(fake_im_r, nrow=4, normalize=True, range=(-1, 1))
                    self.progressive_model.writer.add_image("fake_im_fr", grid, global_iteration)

                    im = im.cpu()
                    grid = make_grid(im, nrow=4, normalize=True, range=(-1, 1))
                    self.progressive_model.writer.add_image("im_real", grid, global_iteration)

                    seg = seg.cpu()
                    im_ = self.progressive_model.color_transfer(seg)

                    grid = make_grid(im_, nrow=4, normalize=True, range=(-1, 1))
                    self.progressive_model.writer.add_image("seg", grid, global_iteration)


                if (iteration + 1) % 100  == 0:
                    print(
                        f"Res {self.progressive_model.generator.res:03d}, {phase.rjust(9)}: Iteration {iteration + 1:05d}/{training_steps:05d}, epoch:{epoch + 1:05d}/{self.opt.epochs:05d}"
                    )

            if epoch % self.opt.save_epoch_freq == 0 or epoch == self.opt.epochs:                
                self.progressive_model.save_model(self.num_semantics, global_iteration, phase)
                self.pix2pix_model.save(str(int(epoch+1)+int(self.opt.which_epoch)))
                util.save_network(self.end2end_model_on_one_gpu.netD2, 'D2', global_iteration+iteration_D2, self.opt)

            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.pix2pix_model.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.pix2pix_model.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in self.optimizer_D2.param_groups:
                param_group['lr'] = new_lr_D

            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr



    def city35to19(self, input, i, j, num_bs, global_iteration):
        old_label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
        new_label = [255,255,255,255,255,255,255,0,1,255,255,2,3,4,255,255,255,5,255,6,7,8,9,10,11,12,13,14,15,255,255,16,17,18,255]
        old2new = {}
        for l in range(35):
            old2new[l] = new_label[l]

        input = input.numpy()
        I = PIL.Image.fromarray(input[0,:,:].astype('uint8'))
        I.save('samples/%s/%s_label35_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration))

        for ids in range(35):
            input= np.where(input==ids, old2new[ids], input)
        input = PIL.Image.fromarray(input[0,:,:].astype('uint8'))
        input.save('samples/%s/%s_label19_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration))


    def test_model(self , global_iteration, n_sample=1000):
        if not os.path.isdir('samples'):
            os.mkdir('samples')
        if not os.path.isdir('samples/%s'%self.opt.name):
            os.mkdir('samples/%s'%self.opt.name)
        self.num_semantics = self.progressive_model.num_semantics
        print('resolution:',int(self.opt.crop_size/self.opt.aspect_ratio))
        self.set_data_resolution(int(self.opt.crop_size/self.opt.aspect_ratio))
        self.pix2pix_model.eval()
        self.progressive_model.generator.eval()
        z_predefined = torch.Tensor(np.load('z_fixed.npy')).cuda()
        num_bs = 1
        for i in range(int(n_sample/num_bs)):
            print(i)
            z = z_predefined[i*num_bs:(i+1)*num_bs,:]
            global_iteration = self.opt.which_iter
            iteration = 0
            dim_ind = 0
            scaling = int(self.opt.crop_size / (self.opt.aspect_ratio * self.progressive_model.generator.res))

            fake = self.progressive_model.generator(z)
            x_fake, x_fake_mc = self.progressive_model.inferenceSampler(fake, scaling, self.progressive_model.num_semantics)
            x_fake_ = self.progressive_model.color_transfer(x_fake)
            fake = x_fake_.cpu()
            
            seg, _, im, _ = self.next_batch_eval()
            seg, seg_mc, im = self.call_next_batch(seg,im)
            seg_color =  self.progressive_model.color_transfer(seg)

            #pix2pix from fake segmenrations

            with torch.no_grad():
                fake_im,_ = self.pix2pix_model.generate_fake(x_fake_mc, im)
            fake_im = fake_im.cpu()
            x_fake = x_fake.cpu()
            for j in range(num_bs):
                self.city35to19(x_fake[j,:,:,:], i,j ,num_bs, global_iteration)

                save_image(fake[j,:,:,:], 'samples/%s/%s_pg_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration),
                             nrow=1, normalize=True, range=(-1,1))

                save_image(fake_im[j,:,:,:], 'samples/%s/%s_spade_fff_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration),
                             nrow=1, normalize=True, range=(-1,1))

            #pix2pix from real segmentations
            with torch.no_grad():
                fake_im, _ = self.pix2pix_model.generate_fake(seg_mc, im)
            fake_im = fake_im.cpu()
            for j in range(num_bs):
                save_image(fake_im[j,:,:,:], 'samples/%s/%s_spade_ffr_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration),
                             nrow=1, normalize=True, range=(-1,1))
            for j in range(num_bs):
                save_image(seg_color[j,:,:,:], 'samples/%s/%s_seg_real_spade_%s_%s.png'%(self.opt.name, i*num_bs+j, self.progressive_model.dim, global_iteration),
                             nrow=1, normalize=True, range=(-1,1))

        fid_real = self.compute_FID(global_iteration, real_fake='fake')
        print('fid fake from fake', fid_real)
        fid = self.compute_FID(global_iteration, real_fake='real')
        print('fid fake from real', fid)


    def compute_FID(self, global_iteration, real_fake='fake', z_fixed=torch.Tensor([])):
        num_ims = 0
        dims = 2048
        dim_ind = 0
        nums_fid = self.opt.nums_fid
        batchsize = 1 # self.batch_size[dim_ind]
        all_reals = np.zeros((int(nums_fid/batchsize)*batchsize,dims))
        all_fakes = np.zeros((int(nums_fid/batchsize)*batchsize,dims))
        scaling = int(self.opt.crop_size / (self.opt.aspect_ratio * self.progressive_model.generator.res))
        upsample = nn.Upsample(scale_factor=scaling, mode='nearest')
        for i in range(int(nums_fid/batchsize)):

            real_segs, _, real_ims, _ = self.next_batch_eval()
            real_ims = Variable(real_ims).cuda()
            real_segs = real_segs.cuda()
            # real_ims = real_ims[:,:,::2,::2]
            with torch.no_grad():
                real_acts = get_activations(real_ims, self.inception_model, batchsize, cuda=True)
            all_reals[i*batchsize:i*batchsize+real_acts.shape[0],:] = real_acts
            if z_fixed.nelement() == 0:
                z = torch.randn(batchsize, 512).cuda()
            else:
                z = z_fixed[i*batchsize:(i+1)*batchsize,:]


            z = torch.randn(batchsize, 512).cuda()
            with torch.no_grad():
                fake = self.progressive_model.generator(z)
            if real_fake=='fake':
                fake = upsample(fake)


                with torch.no_grad():
                    fake_im_f, _ = self.pix2pix_model.generate_fake(fake, real_ims, compute_kld_loss=False)
                    # fake_im_f = fake_im_f[:,:,::2,::2]
                    fake_acts = get_activations(fake_im_f, self.inception_model, batchsize, cuda=True)
                all_fakes[i*batchsize:i*batchsize+fake_acts.shape[0],:] = fake_acts

            else:
                real_segs_mc = torch.FloatTensor(real_segs.size(0), fake.size(1), real_segs.size(2), real_segs.size(3)).zero_().cuda()
                real_segs_mc = real_segs_mc.scatter_(1, real_segs.long(), 1.0)

                with torch.no_grad():
                    fake_im_r, _ = self.pix2pix_model.generate_fake(real_segs_mc, real_ims, compute_kld_loss=False)
                    # fake_im_r = fake_im_r[:,:,::2,::2]
                    fake_acts = get_activations(fake_im_r, self.inception_model, batchsize, cuda=True)
                all_fakes[i*batchsize:i*batchsize+fake_acts.shape[0],:] = fake_acts
            
        fid_eval = calculate_fid_given_acts(all_reals,all_fakes)
        return fid_eval





if __name__ == "__main__":
    opt = parameters().parse()
    torch.backends.cudnn.benchmark = False
    opt.gpu_ids = list(range(int(torch.cuda.device_count())))

    opt.rgb = opt.rgb_seg
    opt.label_nc = opt.label_seg
    print(opt.label_nc)

    if opt.test:
        opt.isTrain = False
        opt.train = False
        if opt.z_notdefined:
            z=np.random.randn(opt.N,512)
            np.save('z_fixed.npz', z)

        ProgressiveTrainer(opt).test_model(opt.which_iter, n_sample=opt.N)
    else:
        opt.isTrain=True
        opt.train=True

        if opt.end2end:
            opt.update_pix2pix = True
            opt.update_pix2pix_w_D2 = True
            opt.update_progan = True
            opt.update_progan_w_D2 = True

        ProgressiveTrainer(opt).run()
