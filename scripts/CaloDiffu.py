import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from models import *


class CaloDiffu(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape,num_batch,name='SGM',config=None, cylindrical = False, R_Z_inputs = False, training_obj = 'noise_pred',
                    cold_diffu = False, E_bins = None, avg_showers = None, std_showers = None):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self._num_batch = num_batch
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = self.config['NSTEPS']
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        if(self.training_obj not in ['noise_pred', 'mean_pred', 'hybrid']):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)


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
        #CHECK OFF BY ONE ERROR ? 

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

        self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, 
                cylindrical =  config.get('CYLINDRICAL', False), data_shape = calo_summary_shape)

        print("\n\n Model: \n")
        summary(self.model, summary_shape)
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]

    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
        out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
        return out


    def compute_loss(self, data, energy, noise = None, loss_type = "l2"):
        if noise is None:
            noise = torch.randn_like(data)
        
        t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()
        x_noisy = self.noise_image(data, t, noise=noise)

        if(self.R_Z_inputs):
            if(data.shape[0] == self._num_batch):
                x_input = torch.cat([x_noisy, self.batch_R_image, self.batch_Z_image], axis = 1)
            else:
                batch_R_image = self.R_image.repeat([data.shape[0], 1,1,1,1])
                batch_Z_image = self.Z_image.repeat([data.shape[0], 1,1,1,1])
                x_input = torch.cat([x_noisy, batch_R_image, batch_Z_image], axis = 1)
        else: x_input = x_noisy

        pred = self.model(x_input, energy, t/self.nsteps)

        if(self.training_obj == 'hybrid'):
            #sample noise from random normal 
            #P_mean = torch.log(self.nsteps/2)
            #P_mean = torch.log(self.nsteps/4)
            #rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            #sigma = (rnd_normal * self.P_std + self.P_mean).exp()

            sigma2 = extract(self.betas, t, data.shape)
            weight = 1. + (1./ sigma2)

            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            c_in = 1. / (sigma2 + 1. ).sqrt()

            pred = c_skip * data + c_out * pred
            target = data
            loss = (weight * ((pred - data) ** 2)).sum() / weight.sum()

            return loss

        elif(self.training_obj == 'noise_pred'):
            target = noise
        elif(self.training_obj == 'mean_pred'):
            target = data




        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()

        return loss
        



    @torch.no_grad()
    def p_sample(self, x, img, cond, t, avg_shower = None, std_shower = None, cold_frac = 0., noise = None):
        #reverse the diffusion process (one step)

        if(noise is None): noise = torch.randn(img.shape, device = x.device)

        betas_t = extract(self.betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, img.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, img.shape)
        posterior_variance_t = extract(self.posterior_variance, t, img.shape)


        #noise_pred = self.model.predict([x, t, cond], batch_size = batch_size)

        sample_algo = 'euler'
        #sample_algo = 'cold'
        #sample_algo = 'stocasticSampler'

        pred = self.model(x, cond, t/self.nsteps)
        if(self.training_obj == 'noise_pred'):
            noise_pred = pred
        elif(self.training_obj == 'mean_pred'):
            noise_pred = (img - sqrt_alphas_cumprod_t * pred)/sqrt_one_minus_alphas_cumprod_t
        elif(self.trainig_obj == 'hybrid'):

            sigma2 = extract(self.betas, t, data.shape)
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            mean_pred = c_skip * img + c_out * pred
            noise_pred = (img - sqrt_alphas_cumprod_t * mean_pred)/sqrt_one_minus_alphas_cumprod_t

        

        if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
            sample_algo = 'cold'
            noise = torch.add(avg_shower, cold_frac * (noise * std_shower))

        if(sample_algo == 'euler'):
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( img - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            if t[0] == 0: return post_mean
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
        elif(sample_algo == 'cold'):
            #Algorithm 2 from cold diffusion paper
            #Work in progress!
            # x_t-1 = x_t - D(x0, t) + D(x0, t-1)

            if t[0] == 0: return post_mean

            x0_pred = (img - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t

            #if(t[0] == 0):  return x0_pred
            algo1 = False

            #algo 1
            out1 = self.noise_image(x0_pred, t-1, noise = noise)
            if(t[0] % 20 == 0):
                print( "ALGO1", torch.mean(out1), torch.mean(self.noise_image(x0_pred, t-1, noise = noise)))
                    #print(torch.mean(out), torch.mean(self.noise_image(x0_pred, t-1, noise = noise)))

            #algo 2
            noise2 = torch.randn(img.shape, device = x.device)
            #if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
            #    noise2 = torch.add(avg_shower, cold_frac * (noise2 * std_shower))
            out2 = img - self.noise_image(x0_pred, t, noise = noise) + self.noise_image(x0_pred, t-1, noise = noise)
            if(t[0] % 20 == 0):
                print("ALGO2", torch.mean(out2), torch.mean(img), torch.mean(self.noise_image(x0_pred, t, noise = noise)), torch.mean(self.noise_image(x0_pred, t-1, noise = noise)))

            out = out2


        print(torch.mean(out))
        return out

    @torch.no_grad()
    def Sample_Cold(self, E, num_steps = 200, cold_frac = 0.):

        device = next(self.parameters()).device
        gen_size = E.shape[0]
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0,gen_size)

        #start from pure noise
        img_start = torch.randn(gen_shape, device=device)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            avg_shower, std_shower = self.lookup_avg_std_shower(E)
            img_start = torch.add(avg_shower, cold_frac * (img_start * std_shower))

        img = noise = img_start
        sampling_routine = 'algo1'

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)

        t = num_steps

        while (t):

            if(self.R_Z_inputs): x_in = torch.cat([img, batch_R_image, batch_Z_image], axis = 1)
            else: x_in = img

            step = torch.full((gen_size,), t - 1, dtype=torch.long, device=img.device)
            noise_pred = self.noise_predictor(x_in, E, step)

            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, step, img.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, step, img.shape)

            x1_bar = (img - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            if sampling_routine == 'ddim':
                x2_bar = noise_pred
            else: x2_bar = noise

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.noise_image(data=xt_bar, noise=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((gen_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.noise_image(data=xt_sub1_bar, noise=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            print(torch.mean(x), torch.mean(img), torch.mean(xt_bar), torch.mean(xt_sub1_bar))
            img = x
            t = t - 1

        return img

        #elif self.sampling_routine == 'x0_step_down':
        #    while (t):
        #        step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
        #        x1_bar = self.denoise_fn(img, step)
        #        x2_bar = noise
        #        if direct_recons == None:
        #            direct_recons = x1_bar

        #        xt_bar = x1_bar
        #        if t != 0:
        #            xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

        #        xt_sub1_bar = x1_bar
        #        if t - 1 != 0:
        #            step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
        #            xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

        #        x = img - xt_bar + xt_sub1_bar
        #        img = x
        #        t = t - 1




    @torch.no_grad()
    def Sample(self, E, num_steps = 200, cold_frac = 0.):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """
        # Full sample (all steps)
        device = next(self.parameters()).device


        gen_size = E.shape[0]
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0,gen_size)

        #start from pure noise
        img_start = torch.randn(gen_shape, device=device)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            avg_shower, std_shower = self.lookup_avg_std_shower(E)
            img_start = torch.add(avg_shower, cold_frac * (img_start * std_shower))


        start = time.time()
        #for i in tqdm(reversed(range(0, num_steps)), desc='sampling loop time step', total=num_steps):
        time_steps = list(range(num_steps))
        time_steps.reverse()

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)

        img = img_start
        fixed_noise = None
        fixed_noise = img_start
        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            if(self.R_Z_inputs): x_in = torch.cat([img, batch_R_image, batch_Z_image], axis = 1)
            else: x_in = img
            img = self.p_sample(x_in, img, E, times, avg_shower = avg_shower, std_shower = std_shower, cold_frac = cold_frac, noise = fixed_noise)

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



            



        
