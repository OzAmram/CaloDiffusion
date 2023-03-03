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
        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        if(not any(is_obj)):
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

        #print("MIN MIDDLE MAX variance", ( self.sqrt_one_minus_alphas_cumprod[0].numpy(),   
            #self.sqrt_one_minus_alphas_cumprod[self.nsteps//2].numpy(), self.sqrt_one_minus_alphas_cumprod[-1].numpy()))
        #exit(1)

        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.R_Z_inputs = config.get('R_Z_INPUT', False)
        in_channels = 1

        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')
        self.R_image, self.Z_image = create_R_Z_image(device, scaled = True)

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

        self.time_embed = config.get("TIME_EMBED", 'sin')
        self.E_embed = config.get("COND_EMBED", 'sin')

        self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, 
                cylindrical =  config.get('CYLINDRICAL', False), data_shape = calo_summary_shape,
                cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

        print("\n\n Model: \n")
        summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]

    def adapt_cold_noise_scale(self, inputEs, shape):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        nbins = len(self.E_bins)
        cold_scales = torch.ones(inputEs.shape[0])
        cold_scales[idxs < nbins/2] = 1.2
        cold_scales[idxs < nbins/4] = 1.3
        cold_scales[idxs < nbins/8] = 1.4
        cold_scales[idxs > 3*nbins/4] = 0.9
        cold_scales[idxs > 7*nbins/8] = 0.8
        return torch.reshape(cold_scales, (shape[0], * ((1,) * (len(shape) - 1)) ))

    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
        out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
        return out


    def compute_loss(self, data, energy, noise = None, t = None, loss_type = "l2"):
        if noise is None:
            noise = torch.randn_like(data)
        
        
        if(t is None):
            t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()

        t_emb = self.do_time_embed(t, self.time_embed)

        x_noisy = self.noise_image(data, t, noise=noise)

        sigma2 = extract(self.betas, t, data.shape)

        if(self.R_Z_inputs):
            if(data.shape[0] == self._num_batch):
                x_input = torch.cat([x_noisy, self.batch_R_image, self.batch_Z_image], axis = 1)
            else:
                batch_R_image = self.R_image.repeat([data.shape[0], 1,1,1,1])
                batch_Z_image = self.Z_image.repeat([data.shape[0], 1,1,1,1])
                x_input = torch.cat([x_noisy, batch_R_image, batch_Z_image], axis = 1)
        else: x_input = x_noisy

        pred = self.model(x_input, energy, t_emb)

        if( 'hybrid' in self.training_obj ):
            #sample noise from random normal 
            #P_mean = torch.log(self.nsteps/2)
            #P_mean = torch.log(self.nsteps/4)
            #rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            #sigma = (rnd_normal * self.P_std + self.P_mean).exp()

            weight = 1. + (1./ sigma2)

            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            c_out = 1./ (1. + 1./sigma2).sqrt()

            #target = (data - c_skip * x_noisy)

            pred = c_skip * x_noisy + c_out * pred
            target = data



            #print(torch.mean(denoised_pred), torch.std(denoised_pred))
            #print(torch.mean(c_out), torch.mean(c_skip))
            #print(torch.mean(target), torch.std(target))


        elif(self.training_obj == 'noise_pred'):
            target = noise
            weight = 1.
        elif(self.training_obj == 'mean_pred'):
            target = data
            weight = 1./ sigma2




        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / weight.sum()
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()

        return loss
        


    def do_time_embed(self, t, embed_type = "identity"):
        if(embed_type == "identity" or embed_type == 'sin'):
            return t
        if(embed_type == "scaled"):
            return t/self.nsteps
        if(embed_type == "sigma"):
            return torch.sqrt(self.betas[t]).to(t.device)

        if(embed_type == "log"):
            return 0.5 * torch.log(self.betas[t]).to(t.device)



    @torch.no_grad()
    def p_sample(self, x, img, E, t, cold_noise_scale = 0., noise = None, sample_algo = 'euler', debug = False):
        #reverse the diffusion process (one step)



        if(noise is None): 
            noise = torch.randn(img.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        betas_t = extract(self.betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, img.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, img.shape)
        posterior_variance_t = extract(self.posterior_variance, t, img.shape)

        t_emb = self.do_time_embed(t, self.time_embed)


        pred = self.model(x, E, t_emb)
        if(self.training_obj == 'noise_pred'):
            noise_pred = pred
            x0_pred = None
        elif(self.training_obj == 'mean_pred'):
            x0_pred = pred
            noise_pred = (img - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
        elif(self.training_obj == 'hybrid'):

            sigma2 = extract(self.betas, t, img.shape)
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            x0_pred = c_skip * img + c_out * pred
            noise_pred = (img - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        


        if(sample_algo == 'euler'):
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( img - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
            if t[0] == 0: out = post_mean

        elif(sample_algo == 'cold_step'):
            post_mean = img - noise_pred * sqrt_one_minus_alphas_cumprod_t
            post_mean = sqrt_recip_alphas_t * ( img - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean
            #out = post_mean + torch.sqrt(posterior_variance_t) * noise 


        elif('cold' in sample_algo):

            if(x0_pred is None):
                x0_pred = (img - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t

            #algo 2 from cold diffu paper
            # x_t-1 = x(t, eps_t) - D(x0, t, eps_t) + D(x0, t-1, eps_t-1)
            #Must use same eps for x_t and D(x0, t), otherwise unstable
            if('cold2' in sample_algo):
                out = img - self.noise_image(x0_pred, t, noise = self.prev_noise) + self.noise_image(x0_pred, t-1, noise = noise)
                self.prev_noise = noise
            else:
            #algo 1
                out = self.noise_image(x0_pred, t-1, noise = noise)
            #print(torch.mean(out), torch.std(out))



        if(debug): 
            if(x0_pred is None):
                x0_pred = (img - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            return out, x0_pred
        return out

    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)

        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)

        if(cold_noise_scale == 'ADAPT'): #special string
            cold_scales = self.adapt_cold_noise_scale(E, avg_shower.shape)
        else: #float
            cold_scales = cold_noise_scale

        return torch.add(avg_shower, cold_scales * (noise * std_shower))




    @torch.no_grad()
    def Sample(self, E, num_steps = 200, cold_noise_scale = 0., sample_algo = 'euler', debug = False, sample_offset = 0):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """

        print("SAMPLE ALGO : %s" % sample_algo)

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
            img_start = self.gen_cold_image(E, cold_noise_scale)


        start = time.time()
        #for i in tqdm(reversed(range(0, num_steps)), desc='sampling loop time step', total=num_steps):
        #time_steps = list(range(num_steps -50))
        time_steps = list(range(num_steps - sample_offset))
        time_steps.reverse()

        batch_R_image = self.R_image.repeat([gen_size, 1,1,1,1]).to(device=device)
        batch_Z_image = self.Z_image.repeat([gen_size, 1,1,1,1]).to(device=device)

        img = img_start
        fixed_noise = None
        print("Start", torch.mean(img_start), torch.std(img_start))
        if('fixed' in sample_algo): 
            print("Fixing noise to constant for sampling!")
            fixed_noise = img_start
        imgs = []
        x0s = []
        self.prev_noise = img_start
        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            if(self.R_Z_inputs): x_in = torch.cat([img, batch_R_image, batch_Z_image], axis = 1)
            else: x_in = img
            out = self.p_sample(x_in, img, E, times, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)
            if(debug): 
                img, x0_pred = out
                imgs.append(img.detach().cpu().numpy())
                x0s.append(x0_pred.detach().cpu().numpy())
            else: img = out

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))
        if(debug):
            return img.detach().cpu().numpy(), imgs, x0s
        else:   
            return img.detach().cpu().numpy()

    
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



            



        
