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
    def __init__(self, data_shape,num_batch,name='SGM',config=None, cylindrical = False, R_Z_inputs = False):
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

        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        if(schedd == "linear"): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif(schedd == "cosine"): 
            print("COSINE SCHEDULE!")
            self.betas = cosine_beta_schedule(self.nsteps)
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        self.alphas = 1. - self.betas


        #precompute useful quantities for training
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

        #shift all elements over by inserting unit value in first place
        alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        print(self.sqrt_one_minus_alphas_cumprod[:10])

        #print("MIN MIDDLE MAX variance", ( self.sqrt_one_minus_alphas_cumprod[0].numpy(),   
            #self.sqrt_one_minus_alphas_cumprod[self.nsteps//2].numpy(), self.sqrt_one_minus_alphas_cumprod[-1].numpy()))
        #exit(1)

        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.R_Z_inputs = config.get('R_Z_INPUT', False)
        in_channels = 1

        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')
        self.R_image, self.Z_image = create_R_Z_image(device)

        if(self.R_Z_inputs):

            self.batch_R_image = self.R_image.repeat([num_batch, 1,1,1,1])
            self.batch_Z_image = self.Z_image.repeat([num_batch, 1,1,1,1])

            in_channels = 3

        layer_sizes = config['LAYER_SIZE_UNET']
        cond_dim = config['COND_SIZE_UNET']

        calo_summary_shape = list(copy.copy(self._data_shape))
        calo_summary_shape.insert(0, self._num_batch)
        calo_summary_shape[1] = in_channels

        calo_summary_shape[0] = 1
        summary_shape = [calo_summary_shape, [1], [1]]

        self.noise_predictor = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, 
                cylindrical =  config.get('CYLINDRICAL', False), data_shape = calo_summary_shape)

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
        if(self.R_Z_inputs):
            if(data.shape[0] == self._num_batch):
                x_input = torch.cat([x_noisy, self.batch_R_image, self.batch_Z_image], axis = 1)
            else:
                batch_R_image = self.R_image.repeat([data.shape[0], 1,1,1,1])
                batch_Z_image = self.Z_image.repeat([data.shape[0], 1,1,1,1])
                x_input = torch.cat([x_noisy, batch_R_image, batch_Z_image], axis = 1)
        else: x_input = x_noisy
        predicted_noise = self.noise_predictor(x_input, energy, random_t)

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
    def p_sample(self, x, img, cond, t):
        #reverse the diffusion process (one step)


        betas_t = extract(self.betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, img.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, img.shape)

        #noise_pred = self.model.predict([x, t, cond], batch_size = batch_size)

        #sample_algo = 'euler'
        sample_algo = 'cold'
        #sample_algo = 'stocasticSampler'

        noise_pred = self.noise_predictor(x, cond, t)
        
        
        # Use results from our model (noise predictor) to predict the mean of posterior distribution
        post_mean = sqrt_recip_alphas_t * ( img - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)

        noise = torch.randn(img.shape, device = x.device)
        posterior_variance_t = extract(self.posterior_variance, t, img.shape)

        if(sample_algo == 'euler'):
            if t[0] == 0: return post_mean
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
        elif(sample_algo == 'cold'):
            #Algorithm 2 from cold diffusion paper
            #Work in progress!
            # x_t-1 = x_t - D(x0, t) + D(x0, t-1)

            x0_pred = (img - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            noise2 = torch.randn(img.shape, device = x.device)

            if(t[0] == 0):  return x0_pred

            #algo 1
            #out = self.noise_image(x0_pred, t-1, noise = noise)

            #algo 2
            out = img - self.noise_image(x0_pred, t, noise = noise2) + self.noise_image(x0_pred, t-1, noise = noise)

            #print(out.shape, img_in.shape, post_mean.shape, noise.shape, posterior_variance_tm1.shape)

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

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)


        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            if(self.R_Z_inputs): x_in = torch.cat([img, batch_R_image, batch_Z_image], axis = 1)
            else: x_in = img
            img = self.p_sample(x_in, img, cond, times)

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))
        return img

    @torch.no_grad()
    def Sample(self, cond, num_steps = 1000):
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

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)


        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            if(self.R_Z_inputs): x_in = torch.cat([img, batch_R_image, batch_Z_image], axis = 1)
            else: x_in = img
            img = self.p_sample(x_in, img, cond, times)

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))
        return img
    
    @torch.no_grad()
    def Sample_v2(self, cond, num_steps = 1000):

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

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)

        train_steps = torch.linspace(0, self.nsteps, self.nsteps + 1)
        C1 = 0.001
        C2 = 0.008
        j0 = 8
        alphas = torch.sin((np.pi/2.0) * train_steps / self.nsteps / (C2 + 1.0)) **2
        Us = [0.]*num_steps
        for j in range(self.nsteps-1, 0, -1):
            Us[j-1] = ( (Us[j]**2 + 1.0) / max((alphas[j-1]/ alphas[j]).item(), C1) - 1.0)**0.5

        Us.reverse()

        print("us", Us[-10:])
        print("self.betas", self.betas[-10:]**0.5)
        print('alphas', 1. - alphas[-10:])
        print('self.alphas_cumprod', self.alphas_cumprod[-10:])
        num_steps = 500
        ts_sample = [np.floor(j0 + (self.nsteps - 1 - j0)/(num_steps -1) * i + 0.5) for i in range(num_steps)]
        ts_sample.reverse()
        print(ts_sample)
        exit(1)



            



        
