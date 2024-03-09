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


class ControlledUNet(nn.Module):
    #Unet plus control net
    def __init__(self, UNet, ControlNet):
        super().__init__()
        self.UNet = UNet
        self.ControlNet = ControlNet
        self.nsteps = self.UNet.nsteps
        self.nets = nn.ModuleList([self.UNet, self.ControlNet])
        #Zero convolutions for downsampling and hidden state layers
        self.control_adds = nn.ModuleList([])
        for i in range(len(self.UNet.model.downs)):
            self.control_adds.append(ScalarAddLayer())

        #For middle hidden state
        self.control_adds.append(ScalarAddLayer())
        self.control_adds.append(ScalarAddLayer())

        
    def denoise(self, x, c_x = None, E =None, sigma=None, model = None, layers = None, layer_sample = False, controls = None):

        if(c_x is None):
            avg_showers, std_showers = self.UNet.lookup_avg_std_shower(E)
            c_x = avg_showers

        #Prepare controlnet inputs
        t_emb = self.UNet.do_time_embed(embed_type = self.UNet.time_embed, sigma = sigma.reshape(-1))
        conds = E
        if(self.ControlNet.layer_cond and layers is not None): conds = torch.cat([E, layers], dim = 1)
        if(self.ControlNet.NN_embed is not None): c_x = self.ControlNet.NN_embed.enc(c_x).to(c_x.device)

        control_hs = self.ControlNet.model.get_hiddens(self.ControlNet.add_RZPhi(c_x), conds, t_emb)

        controls = list(zip(self.control_adds, control_hs))

        out = self.UNet.denoise(x, conds, sigma, controls = controls)
        return out

    def compute_loss(self, data, E, model = None, noise = None, t = None, layers = None, loss_type = "l2", rnd_normal = None, layer_loss = False, scale=1 ):
        if noise is None:
            noise = torch.randn_like(data)

        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        

        rnd_normal = torch.randn((data.size()[0],), device=data.device)
        sigma = (rnd_normal * self.UNet.P_std + self.UNet.P_mean).exp().reshape(const_shape)

        x_noisy = data + sigma * noise
        sigma2 = sigma**2

        weight = 1.

        x0_pred = self.denoise(x_noisy, E=E, sigma=sigma, model = model, layers = layers)

        training_obj = self.UNet.training_obj

        if('hybrid' in training_obj ):
            weight = torch.reshape(1. + (1./ sigma2), const_shape)
            target = data
            pred = x0_pred

        elif('noise_pred' in training_obj):
            pred = (data - x0_pred)/sigma
            target = noise
            weight = 1.
        elif('mean_pred' in training_obj):
            target = data
            weight = 1./ sigma2
            pred = x0_pred
            
        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
            
        else:
            print(loss_type)
            raise NotImplementedError()


        return loss

    def __call__(self, x, **kwargs):
        return self.denoise(x, **kwargs)

    def Sample(self, E, layer_sample = False, model = None, **kwargs):

        if(layer_sample): 
            return self.UNet.Sample(E, layer_sample = True, model = model, **kwargs)

        else:
            return self.UNet.Sample(E, model = self, **kwargs)







