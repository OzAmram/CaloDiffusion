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
    def __init__(self, data_shape = None, config=None, R_Z_inputs = False, training_obj = 'noise_pred', nsteps = 400,
                    cold_diffu = False, E_bins = None, avg_showers = None, std_showers = None, NN_embed = None, layer_model = None):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = nsteps
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        self.shower_embed = self.config.get('SHOWER_EMBED', '')
        self.fully_connected = ('FCN' in self.shower_embed)
        self.NN_embed = NN_embed
        self.layer_model = layer_model

        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        if(not any(is_obj)):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)


        if config is None:
            raise ValueError("Config file not given")
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        self.verbose = 1

        
        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')

        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        
        if("linear" in schedd): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        else: self.betas = cosine_beta_schedule(self.nsteps)

        if("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.2
            self.P_std = 1.2

        #precompute useful quantities for training
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

        #shift all elements over by inserting unit value in first place
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.time_embed = config.get("TIME_EMBED", 'sin')
        self.E_embed = config.get("COND_EMBED", 'sin')
        cond_dim = config['COND_SIZE_UNET']
        layer_sizes = config['LAYER_SIZE_UNET']
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)

        #layer energies in conditional info or not
        if('layer' in config.get('SHOWERMAP', '')): 
            self.layer_cond = True
            #gen energy + total deposited energy + layer energy fractions
            cond_size = 2 + config['SHAPE_PAD'][2]
        else: 
            self.layer_cond = False
            cond_size = 1




        if(self.fully_connected):
            #fully connected network architecture
            self.model = FCN(cond_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'],
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

            self.R_Z_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1], [1]]


        else:
            RZ_shape = config['SHAPE_PAD'][1:]

            self.R_Z_inputs = config.get('R_Z_INPUT', False)
            self.phi_inputs = config.get('PHI_INPUT', False)

            in_channels = 1

            self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = RZ_shape)
            self.phi_image = create_phi_image(device, shape = RZ_shape)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [[cond_size]], [1]]


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), cond_size = cond_size, time_embed = (self.time_embed == 'sin')  )

        print("\n\n Model: \n")
        #summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)

    def add_RZPhi(self, x):
        if(len(x.shape) < 3): return x
        cats = [x]
        const_shape = (x.shape[0], *((1,) * (len(x.shape) - 1)))
        if(self.R_Z_inputs):

            batch_R_image = self.R_image.repeat(const_shape).to(device=x.device)
            batch_Z_image = self.Z_image.repeat(const_shape).to(device=x.device)

            cats+= [batch_R_image, batch_Z_image]
        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat(const_shape).to(device=x.device)

            cats += [batch_phi_image]

        if(len(cats) > 1):
            return torch.cat(cats, axis = 1)
        else: 
            return x
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]






    def compute_loss(self, data, E, noise = None, t = None, layers = None, loss_type = "l2", rnd_normal = None, layer_loss = False ):
        if noise is None:
            noise = torch.randn_like(data)

        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        

        if(self.discrete_time): 
            if(t is None): t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()

            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            sigma = sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t

        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)

            sigma = (rnd_normal * self.P_std + self.P_mean).exp().reshape(const_shape)

        x_noisy = data + sigma * noise
        sigma2 = sigma**2

        weight = 1.

        model = self.model if not layer_loss else self.layer_model
        x0_pred = self.denoise(x_noisy, E, sigma, model = model, layers = layers)

        if('hybrid' in self.training_obj ):
            weight = torch.reshape(1. + (1./ sigma2), const_shape)
            target = data
            pred = x0_pred

        elif('noise_pred' in self.training_obj):
            pred = (data - x0_pred)/sigma
            target = noise
            weight = 1.
        elif('mean_pred' in self.training_obj):
            target = data
            weight = 1./ sigma2
            pred = x0_pred


        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()


        return loss
        


    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        if(sigma is None): sigma = self.sqrt_one_minus_alphas_cumprod[t] /self.sqrt_alphas_cumprod[t]

        if(embed_type == "identity" or embed_type == 'sin'):
            return t
        if(embed_type == "scaled"):
            return t/self.nsteps
        if(embed_type == "sigma"):
            #return torch.sqrt(self.betas[t]).to(t.device)
            my_sigma = sigma / (1 + sigma**2).sqrt()
            return my_sigma
        if(embed_type == "log"):
            return 0.5 * torch.log(sigma)

    def pred(self, x, E, t_emb, model = None, layers = None):
        if(model is None): model = self.model

        if(self.NN_embed is not None): x = self.NN_embed.enc(x).to(x.device)
        if(self.layer_cond and layers is not None): E = torch.cat([E, layers], dim = 1)
        out = model(self.add_RZPhi(x), E, t_emb)
        if(self.NN_embed is not None): out = self.NN_embed.dec(out).to(x.device)
        return out

    def denoise(self, x, E, sigma, model = None, layers = None):
        t_emb = self.do_time_embed(embed_type = self.time_embed, sigma = sigma.reshape(-1))
        sigma = sigma.reshape(-1, *(1,)*(len(x.shape)-1))
        c_in = 1 / (sigma**2 + 1).sqrt()

        pred = self.pred(x * c_in, E, t_emb, model = model, layers = layers)

        if('noise_pred' in self.training_obj):
            return (x - sigma * pred)

        if('mean_pred' in self.training_obj):
            return pred
        elif('hybrid' in self.training_obj):

            sigma2 = sigma**2
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            return (c_skip * x + c_out * pred)



    def edm_sampler( self, x, E, layers = None, sample_algo = 'euler', randn_like=torch.randn_like, num_steps=400, sigma_min=0.002, sigma_max=1, rho=7,
        S_churn=0, S_min=0, S_max=1.0, S_noise=1,sample_offset = 0, orig_schedule = False, model = None):
        #EDM sampler (and variations), adapted from  https://github.com/NVlabs/edm


        gen_size = x.shape[0]

        # Time step discretization from edm paper
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        t_steps = t_steps[sample_offset:]


        if(orig_schedule): #use steps from original iDDPM schedule 
            M = self.nsteps # num steps trained with
            C_2 = 0.0008
            C_1 = 0.001
            u = torch.zeros(M + 1, dtype=torch.float64, device=x.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=x.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            t_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)] 


        #print(t_steps)
    
        # Main sampling loop.
        x_next = x.to(torch.float32) * t_steps[0]
        t_next = t_steps[0]
        xs = []
        x0s = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
            denoised = self.denoise(x_hat, E, t_hat_full, layers = layers, model = model).to(torch.float32) 
           
            d_cur = (x_hat - denoised)/t_hat
            h = t_next - t_hat
            x_prime = x_hat +  h * d_cur
            t_prime = t_hat +  h

            #euler step
            if 'euler' in sample_algo or i == num_steps - sample_offset - 1:
                x_next = x_hat + h * d_cur
            else:
                # 2nd order correction.
                assert ('heun' in sample_algo or 'edm' in sample_algo)
                t_prime_full = torch.full((gen_size,), t_prime, device = x.device)
                denoised = self.denoise(x_prime, E, t_prime_full,layers = layers, model = model).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)

            #print(i, torch.mean(x_cur), torch.mean(denoised))
            xs.append(x_cur)
            x0s.append(denoised)


        return x_next, xs, x0s



    @torch.no_grad()
    def p_sample(self, x, E, t, layers = None, cold_noise_scale = 0., noise = None, sample_algo = 'ddpm', debug = False, model = None):
        #One step of ddpm or ddim sampler
        #Using formalism / notation of EDM paper

        if(noise is None): 
            noise = torch.randn(x.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        #print(t, self.sqrt_one_minus_alphas_cumprod[t])
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        sigma = sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t

        x0_pred = self.denoise(x, E, sigma, layers = layers, model = model)
        noise_pred = (x - x0_pred)/sigma

        if(sample_algo == 'consis'):
            sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, t-1, x.shape) / extract(self.sqrt_alphas_cumprod, t-1, x.shape) if t[0] > 0 else 0.
            #print(sigma[0], sigma_next[0])
            out = x0_pred + sigma_next * noise


        else:
            if(sample_algo == 'ddpm'):
                #using result from ddim paper, which reformulates the ddpm sampler in their notation (See Eq. 12 and sigma definition)
                ddim_eta = 1.0
            else:
                #pure ddim (no stochasticity)
                ddim_eta = 0.0

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_prev = extract(self.alphas_cumprod_prev, t, x.shape)

            denom = extract(self.sqrt_alphas_cumprod, torch.maximum(t-1, torch.zeros_like(t)), x.shape)

            ddim_sigma = ddim_eta * (( (1 - alpha_prev) / (1 - alpha)) * (1 - alpha / alpha_prev))**0.5
            num = (1. - alpha_prev - ddim_sigma**2).sqrt()
            sigma_prev = num / denom


            dir_xt = sigma_prev * noise_pred

            #don't step for t= 0
            mask = (t > 0).reshape(-1, *((1,) *(len(x.shape) - 1)))


            out = x0_pred + mask * sigma_prev * noise_pred + ddim_sigma * noise / denom


        #elif('cold' in sample_algo):

        #    if(x0_pred is None):
        #        x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t

        #    #algo 2 from cold diffu paper
        #    # x_t-1 = x(t, eps_t) - D(x0, t, eps_t) + D(x0, t-1, eps_t-1)
        #    #Must use same eps for x_t and D(x0, t), otherwise unstable
        #    if('cold2' in sample_algo):
        #        out = x - self.noise_image(x0_pred, t, noise = self.prev_noise) + self.noise_image(x0_pred, t-1, noise = noise)
        #        self.prev_noise = noise
        #    else:
        #    #algo 1
        #        out = self.noise_image(x0_pred, t-1, noise = noise)
        #    #print(torch.mean(out), torch.std(out))

        #print(t[0], torch.mean(out), torch.mean(x0_pred))



        if(debug): return out, x0_pred
        return out

    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)
        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)
        cold_scales = cold_noise_scale
        return torch.add(avg_shower, cold_scales * (noise * std_shower))

    def dd_sampler(self, x_start, E, layers = None, num_steps = 400, sample_offset = 0, sample_algo = 'ddpm', model = None, debug = False):
        #ddpm and ddim samplers

        gen_size = E.shape[0]
        device = x_start.device 

        fixed_noise = None
        if('fixed' in sample_algo): 
            print("Fixing noise to constant for sampling!")
            fixed_noise = x_start
        xs = []
        x0s = []
        self.prev_noise = x_start

        time_steps = list(range(0, num_steps - sample_offset))
        time_steps.reverse()

        #scale starting point to appropriate noise level
        sigma_start = self.sqrt_one_minus_alphas_cumprod[time_steps[0]] / self.sqrt_alphas_cumprod[time_steps[0]]
        x = x_start * sigma_start


        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            out = self.p_sample(x, E, times, layers = layers, noise = fixed_noise, sample_algo = sample_algo, debug = debug, model = model)
            if(debug): 
                x, x0_pred = out
                xs.append(x.detach().cpu().numpy())
                x0s.append(x0_pred.detach().cpu().numpy())
            else: x = out
        return x, xs, x0s

    def consis_sampler(self, x_start, E, num_steps = 1, layers = None, model = None):

        sigma_min = 0.002
        sigma_max = 500.0
        rho=7.0

        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_start.device)
        if(num_steps > 1):
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            t_steps = torch.tensor([sigma_max])
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        #t_steps = torch.tensor([500., 10., 1., 0.5, 0.1, 0.01, 0.])
        imax = num_steps

        time_steps = list(range(0, num_steps))
        x = x_start * sigma_max
        
        x0s = []
        xs = []


        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            t_cur_full = torch.full((x_start.shape[0],), t_cur, device=x.device)
            x0 = self.denoise(x, E, t_cur_full, layers = layers, model = model).to(torch.float32) 

            t_next = torch.clip(t_next, sigma_min, sigma_max)
            x = x0 + (t_next > 0 ) * torch.randn_like(x) * torch.sqrt(t_next**2 - sigma_min**2)
            x0s.append(x0)
            xs.append(x)

        return x,xs,x0






    @torch.no_grad()
    def Sample(self, E, layers = None, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, model = None, gen_shape = None):
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


        if(gen_shape is None):
            gen_size = E.shape[0]
            gen_shape = list(copy.copy(self._data_shape))
            gen_shape.insert(0,gen_size)

        #start from pure noise
        x_start = torch.randn(gen_shape, device=device)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)


        if('euler' in sample_algo or 'edm' in sample_algo or 'heun' in sample_algo):
            S_churn = 40 if (sample_algo == 'edm' or 'noise' in sample_algo) else 0  #Number of steps to 'reverse' to add back noise
            S_min = 0.01
            S_max = 50
            S_noise = 1.003
            sigma_min = 0.002
            #sigma_max = 500.0
            sigma_max = 80.0
            orig_schedule = False

            x,xs, x0s = self.edm_sampler(x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max, 
                    S_churn = S_churn, S_min = S_min, S_max = S_max, S_noise = S_noise, sample_offset = sample_offset, orig_schedule = orig_schedule, model = model)

        elif('consis' in sample_algo):
            x,xs, x0s = self.consis_sampler(x_start, E, layers = layers, num_steps = num_steps, model = model)
        else:
            x, xs, x0s = self.dd_sampler(x_start, E, layers = layers, num_steps = num_steps, sample_offset = sample_offset, sample_algo = sample_algo, debug = debug, model = model)

        if(debug):
            return x.detach().cpu().numpy(), xs, x0s
        else:   
            return x.detach().cpu().numpy()


    
        
