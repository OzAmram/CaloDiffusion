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


class ConsistencyModel(nn.Module):
    "Consistency model a al  https://arxiv.org/pdf/2303.01469.pdf"
    def __init__(self, config, diffu_model = None):
        self.CaloDiffuModel = diffu_model
        self.add_RZ = self.CaloDiffuModel.add_RZ


        self.R_Z_inputs = config.get('R_Z_INPUT', False)
        in_channels = 1

        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')
        self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = self._data_shape)

        if(self.R_Z_inputs):

            self.batch_R_image = self.R_image.repeat([num_batch, 1,1,1,1])
            self.batch_Z_image = self.Z_image.repeat([num_batch, 1,1,1,1])

            in_channels = 3

        layer_sizes = config['LAYER_SIZE_UNET']
        cond_dim = config['COND_SIZE_UNET']

        self.time_embed = config.get("TIME_EMBED", 'sin')
        self.E_embed = config.get("COND_EMBED", 'sin')


        self._data_shape = config['SHAPE_PAD'][1:]
        calo_summary_shape = list(copy.copy(self._data_shape))
        calo_summary_shape.insert(0, self._num_batch)
        calo_summary_shape[1] = in_channels

        calo_summary_shape[0] = 1

        self.sigma_min = self.CaloDiffuModel.sqrt_one_minus_alphas_cumprod[-1]

        self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, 
                cylindrical =  config.get('CYLINDRICAL', False), data_shape = calo_summary_shape,
                cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )


        def pred(self, x, E, t):

            sigma = extract(self.CaloDiffuModel.sqrt_one_minus_alphas_cumprod, t, data.shape)
            t_emb = self.do_time_embed(t, self.time_embed, sigma)

            pred = self.model(self.add_RZ(x_input), energy, t_emb)


            sigma2 = (sigma - self.sigma_min)**2
            c_skip = torch.reshape(1. / (sigma2 + 1.), (x.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (x.shape[0], 1,1,1,1))

            return c_skip * x + c_out * pred

        def compute_loss(x, noise, E, t, ema_model, sample_algo = 'ddpm'):   

            x_noise = noise_image(x, t)

            #denoise 1 step using diffusion model
            x_prev = self.CaloDiffuModel.p_sample(x_noise, E, t, sample_algo = sample_algo)

            #predict ema model on denoised x
            x0_ema = ema_model.pred(x_prev, E,t)
            #predict model on x
            x0 = model.pred(x_noise, E,t)

            loss = torch.nn.functional.mse_loss(x0_ema, x0)
            return loss



