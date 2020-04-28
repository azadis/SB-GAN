from abc import ABCMeta
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(metaclass=ABCMeta):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    @abstractmethod
    def forward(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def generator_loss(self, x_fake: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def discriminator_loss(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> torch.Tensor:
        pass


class GANHingeLoss(GANLoss):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    def forward(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)

        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(d_real, d_fake)

        return gen_loss, dis_loss

    __call__ = forward

    def generator_loss(self, x_fake: torch.Tensor) -> torch.Tensor:
        d_fake = self.discriminator(x_fake)
        return self.generator_loss_logits(d_fake)

    def generator_loss_logits(self, d_fake: torch.Tensor) -> torch.Tensor:
        return -d_fake.mean()

    def discriminator_loss(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> torch.Tensor:
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)
        return self.discriminator_loss_logits(d_real, d_fake)

    def discriminator_loss_logits(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        real_loss = F.relu(1 - d_real).mean()
        fake_loss = F.relu(1 + d_fake).mean()
        return (real_loss + fake_loss) / 2


class ImprovedWGANLoss(GANLoss):
    def __init__(self, discriminator, lambda_=10.0):
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def gradient_penalty(self, x_real, x_fake, forward=None):
        if forward is None:
            forward = self.discriminator
        n = x_real.size(0)
        device = x_real.device

        alpha = torch.rand(n)
        alpha = alpha.to(device)
        alpha = alpha[:, None, None, None]

        interpolates = alpha * x_real.detach() + (1 - alpha) * x_fake.detach()
        interpolates.requires_grad = True
        dis_interpolates = forward(interpolates)


        grad_outputs = torch.ones_like(dis_interpolates).to(device)
        grad = torch.autograd.grad(
            outputs=dis_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.size(0), -1)

        penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def forward(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)

        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(x_real, x_fake, d_real, d_fake)

        return gen_loss, dis_loss

    def generator_loss(self, x_fake: torch.Tensor) -> torch.Tensor:
        d_fake = self.discriminator(x_fake)
        return self.generator_loss_logits(d_fake)

    def generator_loss_logits(self, d_fake: torch.Tensor) -> torch.Tensor:
        return -d_fake.mean()

    def discriminator_loss(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> torch.Tensor:
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)
        return self.discriminator_loss_logits(x_real, x_fake, d_real, d_fake)

    def discriminator_loss_logits(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        forward=None
    ) -> torch.Tensor:
        grad_penalty = self.gradient_penalty(x_real, x_fake, forward=forward)
        return d_fake.mean() - d_real.mean() + self.lambda_ * grad_penalty



    __call__ = forward


class DiscreteImprovedWGANLoss(torch.autograd.Function):
    def __init__(self, discriminator, lambda_=10.0):
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def sample(self, fake:torch.Tensor):
        fake_cumsum = torch.cumsum(fake, 1)
        alpha = torch.rand(fake.size(0), fake.size(2), fake.size(3)).unsqueeze(1)
        alpha = alpha.repeat(1, fake.size(1), 1, 1).cuda()
        sample_ind = torch.sum((fake_cumsum < alpha)*1, dim=1,keepdim=True)
        sample_ind = sample_ind.type(torch.cuda.LongTensor)
        sample_fake = fake[sample_ind]
        return sample_fake

    @staticmethod
    def forward(ctx, fake: torch.Tensor):
        x_fake = self.sample(fake)
        d_fake = self.discriminator(x_fake)
        ctx.save_for_backward(fake, x_fake)
        return d_fake

    def loss(self, d_fake: torch.Tensor):
        return -d_fake.mean()

    @staticmethod
    def backward(ctx):
        fake, x_fake = ctx.saved_tensors

        #d_fake(\bar{x}) * \delta log p(\bar{x})
        log_softmax_grad = torch.log(fake).grad
        grad = d_fake * log_softmax_grad[x_fake]
        return -grad.mean()


class GANSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, generator=torch.Tensor([]), disc=False, discriminator=torch.Tensor([]), interpolate=False, alpha=0):
        if interpolate:
            fake = generator.interpolate(z, alpha)
        else:
            fake = generator(z)
        print('z, fake', z.requires_grad, fake.requires_grad)
        fake_cumsum = torch.cumsum(fake, 1)
        rand_mat = torch.rand(fake.size(0), fake.size(2), fake.size(3)).unsqueeze(1)
        rand_mat = rand_mat.repeat(1, fake.size(1), 1, 1).cuda()
        sample_ind = torch.sum((fake_cumsum < rand_mat)*1, dim=1,keepdim=True)
        sample_ind = sample_ind.type(torch.cuda.LongTensor)
        x_fake = torch.gather(fake, dim=1, index=sample_ind)
        # x_fake = fake[sample_ind]

        if disc:
            if interpolate:
                d_fake = discriminator.interpolate(x_fake, alpha)
            else:
                d_fake = discriminator(x_fake)

            print('d_fake', d_fake.requires_grad)
            ctx.save_for_backward(fake, x_fake, d_fake)
            return d_fake
        else:
            ctx.save_for_backward(fake, x_fake)
            return x_fake
    
    @staticmethod
    def backward(ctx, grad_output):
        fake, x_fake, d_fake = ctx.saved_tensors
        print('b fake', fake.requires_grad)
        print(fake.size())
        #d_fake(\bar{x}) * \delta log p(\bar{x})
        log_softmax_grad = torch.log(fake).grad
        grad = d_fake * torch.gather(log_softmax_grad, dim=1, index=x_fake)
        grad = grad_output * grad
        print(grad.size())
        return grad






