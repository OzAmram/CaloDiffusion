import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from models import *

#https://huggingface.co/blog/annotated-diffusion


class CaloDiffu(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape,num_batch,name='SGM',config=None, cylindrical = False):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self._num_batch = num_batch
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = self.config['NSTEPS']


        if config is None:
            raise ValueError("Config file not given")
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        self.verbose = 1

        

        #Convolutional model for 3D images and dense for flatten inputs

        #nested arrays to get shapes right
        self.beta_start = 0.0001
        self.beta_end = 0.02

        #linear schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        self.alphas = 1. - self.betas


        #precompute useful quantities for training
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

        #shift all elements over by inserting unit value in first place
        alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        layer_sizes = config['LAYER_SIZE_UNET']
        cond_dim = config['COND_SIZE_UNET']

        self.noise_predictor = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = 1, layer_sizes = layer_sizes, cylindrical = cylindrical)

            
        calo_summary_shape = list(copy.copy(self._data_shape))
        calo_summary_shape.insert(0, 1)
        summary_shape = [calo_summary_shape, [1], [1]]
        print("\n\n Noise predictor model: \n")
        summary(self.noise_predictor, summary_shape)
    
    
    def noise_image(self, data, random_t, noise = None):
        if(noise is None): noise = torch.randn_like(data)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, random_t, data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, random_t, data.shape)
        out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
        return out

    def compute_loss(self, data, energy, random_t, noise = None, loss_type = "l2"):
        if noise is None:
            noise = torch.randn_like(data)

        x_noisy = self.noise_image(data, random_t, noise=noise)
        predicted_noise = self.noise_predictor(x_noisy, energy, random_t)

        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
        



    @torch.no_grad()
    def p_sample(self, x, cond, t):
        #reverse the diffusion process (one step)


        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        #noise_pred = self.model.predict([x, t, cond], batch_size = batch_size)
        noise_pred = self.noise_predictor(x, cond, t)
        
        
        # Use our model (noise predictor) to predict the mean of denoised
        model_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)

        #all t's are the same
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            out = model_mean + torch.sqrt(posterior_variance_t) * noise 
            return out


    @torch.no_grad()
    def Sample(self, cond, num_steps = 200):
        """Generate samples from diffusion model.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """
        # Full sample (all steps)
        device = next(self.parameters()).device


        gen_size = cond.shape[0]
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0,gen_size)


        img = torch.randn(gen_shape, device=device)
        imgs = []

        start = time.time()
        #for i in tqdm(reversed(range(0, num_steps)), desc='sampling loop time step', total=num_steps):
        time_steps = list(range(num_steps))
        time_steps.reverse()

        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, times)

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))
        return img


