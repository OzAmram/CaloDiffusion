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
        self.discrete_time = True

        
        if("linear" in schedd): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif("cosine" in schedd): 
            self.betas = cosine_beta_schedule(self.nsteps)
        elif("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.5
            self.P_std = 1.5
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if(self.discrete_time):
            #precompute useful quantities for training
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

            #shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

            self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

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

    def add_RZ(self, x):
        if(self.R_Z_inputs):
            batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            return torch.cat([x, batch_R_image, batch_Z_image], axis = 1)
        else:
            return x
            
    
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

        if(self.discrete_time):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("NON DISCRETE TIME BAD")
            exit(1)


    def compute_loss(self, data, energy, noise = None, t = None, loss_type = "l2", rnd_normal = None):
        if noise is None:
            noise = torch.randn_like(data)
        
        


        if(self.discrete_time): 
            if(t is None): t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()
            x_noisy = self.noise_image(data, t, noise=noise)
            sigma = None
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)**2
        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            x_noisy = data + torch.reshape(sigma, (data.shape[0], 1,1,1,1)) * noise
            sigma2 = sigma**2



        t_emb = self.do_time_embed(t, self.time_embed, sigma)

        pred = self.model(self.add_RZ(x_noisy), energy, t_emb)

        weight = 1.
        if('hybrid' in self.training_obj ):

            c_skip = torch.reshape(1. / (sigma2 + 1.), (data.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (data.shape[0], 1,1,1,1))
            weight = torch.reshape(1. + (1./ sigma2), (data.shape[0], 1,1,1,1))

            target = (data - c_skip * x_noisy)/c_out


            pred = c_skip * x_noisy + c_out * pred
            target = data

        elif('noise_pred' in self.training_obj):
            target = noise
            weight = 1.
        elif('mean_pred' in self.training_obj):
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
        


    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        if(self.discrete_time):
            if(sigma is None): sigma = self.sqrt_one_minus_alphas_cumprod[t]

            if(embed_type == "identity" or embed_type == 'sin'):
                return t
            if(embed_type == "scaled"):
                return t/self.nsteps
            if(embed_type == "sigma"):
                #return torch.sqrt(self.betas[t]).to(t.device)
                return sigma.to(t.device)
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
        else:
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
            else:
                return sigma


    def edm_sampler( self, x, E, sample_algo = 'euler', randn_like=torch.randn_like, num_steps=400, sigma_min=0.002, sigma_max=10, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
        # Adjust noise levels based on what's supported by the network.

        #sigma_min = max(sigma_min, net.sigma_min)
        #sigma_max = min(sigma_max, net.sigma_max)

        gen_size = x.shape[0]

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = x.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.

            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
            denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur



            # Apply 2nd order correction.
            if (sample_algo == 'edm') and (i < num_steps - 1):
                t_next_full = torch.full((gen_size,), t_next, device=x.device)
                denoised = self.denoise(x_next, E, t_next_full).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


    def denoise(self, x, E,t_emb):
        pred = self.model(self.add_RZ(x), E, t_emb)
        if('mean_pred' in self.training_obj):
            return pred
        elif('hybrid' in self.training_obj):

            sigma2 = (t_emb**2).reshape(-1,1,1,1,1)
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            return c_skip * x + c_out * pred


    @torch.no_grad()
    def p_sample(self, x, E, t, cold_noise_scale = 0., noise = None, sample_algo = 'ddpm', debug = False):
        #reverse the diffusion process (one step)



        if(noise is None): 
            noise = torch.randn(x.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        t_emb = self.do_time_embed(t, self.time_embed)


        pred = self.model(self.add_RZ(x), E, t_emb)
        if('noise_pred' in self.training_obj):
            noise_pred = pred
            x0_pred = None
        elif('mean_pred' in self.training_obj):
            x0_pred = pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
        elif('hybrid' in self.training_obj):

            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**2
            #sigma2 = extract(self.betas, t, x.shape)**2
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            x0_pred = c_skip * x + c_out * pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        


        if(sample_algo == 'ddpm'):
            # Sampling algo from https://arxiv.org/abs/2006.11239
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
            if t[0] == 0: out = post_mean

        elif(sample_algo == 'ddim'):
            if(x0_pred is None): x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            if t[0] == 0: out = x0_pred
            else:
                t_next = t-1
                sqrt_alphas_cumprod_t_next = extract(self.sqrt_alphas_cumprod, t_next, x.shape)

                c1 =  torch.sqrt(1. - sqrt_alphas_cumprod_t_next **2 - posterior_variance_t **2)
                out = sqrt_alphas_cumprod_t_next * x0_pred + c1 * noise_pred + sqrt_alphas_cumprod_t * noise
                print(torch.mean(out), torch.mean(x0_pred), torch.mean(c1 * noise_pred), torch.mean(sqrt_alphas_cumprod_t * noise))

        elif(sample_algo == 'cold_step'):
            post_mean = x - noise_pred * sqrt_one_minus_alphas_cumprod_t
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean
            #out = post_mean + torch.sqrt(posterior_variance_t) * noise 


        elif('cold' in sample_algo):

            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t

            #algo 2 from cold diffu paper
            # x_t-1 = x(t, eps_t) - D(x0, t, eps_t) + D(x0, t-1, eps_t-1)
            #Must use same eps for x_t and D(x0, t), otherwise unstable
            if('cold2' in sample_algo):
                out = x - self.noise_image(x0_pred, t, noise = self.prev_noise) + self.noise_image(x0_pred, t-1, noise = noise)
                self.prev_noise = noise
            else:
            #algo 1
                out = self.noise_image(x0_pred, t-1, noise = noise)
            #print(torch.mean(out), torch.std(out))



        if(debug): 
            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
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
    def Sample(self, E, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, sample_step = 1):
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
        x_start = torch.randn(gen_shape, device=device)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)


        start = time.time()

        if(sample_algo == 'euler' or sample_algo == 'edm'):
            S_churn = 40
            S_min = 0.01
            S_max = 50
            S_noise = 1.005
            sigma_min = 0.002
            sigma_max = 1.0

            x = self.edm_sampler(x_start,E, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max, 
                    S_churn = S_churn, S_min = S_min, S_max = S_max, S_noise = S_noise)

        else:
            x = x_start
            fixed_noise = None
            if('fixed' in sample_algo): 
                print("Fixing noise to constant for sampling!")
                fixed_noise = x_start
            xs = []
            x0s = []
            self.prev_noise = x_start

            time_steps = list(range(0, num_steps - sample_offset, sample_step))
            time_steps.reverse()

            for time_step in time_steps:      
                times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
                out = self.p_sample(x, E, times, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)
                if(debug): 
                    x, x0_pred = out
                    xs.append(x.detach().cpu().numpy())
                    x0s.append(x0_pred.detach().cpu().numpy())
                else: x = out

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))
        if(debug):
            return x.detach().cpu().numpy(), xs, x0s
        else:   
            return x.detach().cpu().numpy()

    
        
