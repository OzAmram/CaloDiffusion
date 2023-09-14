import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from models import *
from CaloDiffu import *

from ema_pytorch import EMA


class ConsistencyModel(nn.Module):
    "Consistency model a al  https://arxiv.org/pdf/2303.01469.pdf"
    def __init__(self, config = None, diffu_model = None):
        super(ConsistencyModel, self).__init__()
        self.dModel = diffu_model

        self.nsteps = diffu_model.nsteps
        self.discrete_time = diffu_model.discrete_time


        self.model = copy.deepcopy(self.dModel.model)
        self.ema_model = EMA(self.model, 
                            beta = 0.99,              # exponential moving average factor
                            update_after_step = 10000,    # only after this number of .update() calls will it start updating
                            update_every = 1,          # how often to actually update, to save on compute (updates every 10th .update() call)
                                    )



    def compute_loss(self, x, E, t = None, noise = None, sample_algo = 'ddim'):   

        #sample noisy example
        if noise is None:
            noise = torch.randn_like(x)
        #if(self.discrete_time): 
        if(t is None): t = torch.randint(0, self.nsteps, (x.size()[0],), device=x.device).long()

        sqrt_alphas_cumprod_t = extract(self.dModel.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.dModel.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sigma = sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t

        #else:
        #    rnd_normal = torch.randn((x.size()[0],), device=x.device)
        #    sigma = (rnd_normal * self.dModel.P_std + self.dModel.P_mean).exp().reshape(x.shape[0], 1,1,1,1)

        x_noisy = x + sigma * noise
        sigma2 = sigma**2


        #TODO Figure out whats causing Nan's...
        with torch.no_grad():
            #denoise 1 step using fixed diffusion model
            x_prev = self.dModel.p_sample(x_noisy, E, t, sample_algo = sample_algo)

            #predict using ema model on one-step denoised x
            x0_ema = self.dModel.denoise(x_prev, E,t, model = self.ema_model)

        #predict using model on noisy x
        x0 = self.dModel.denoise(x_noisy, E,t, model = self.model)
        print(t)
        print(torch.mean(x_prev), torch.mean(x0_ema), torch.mean(x0))

        loss = torch.nn.functional.mse_loss(x0_ema, x0)
        print(torch.mean(loss))
        return loss



