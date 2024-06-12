import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from sampling import *
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
        self.consis_nsteps = self.config.get('CONSIS_NSTEPS', 100)
        self.fully_connected = ('FCN' in self.shower_embed)
        self.NN_embed = NN_embed
        self.layer_model = layer_model

        supported = ['noise_pred', 'mean_pred', 'hybrid', 'minsnr']
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

        

        if("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.5

        self.set_sampling_steps(nsteps)


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
            cond_size = 2 + config['SHAPE_FINAL'][2]
        else: 
            self.layer_cond = False
            cond_size = 1
        if(config.get("HGCAL", False)): 
            cond_size += 2
        print("Cond size %i" % cond_size)




        if(self.fully_connected):
            #fully connected network architecture
            self.model = ResNet(cond_emb_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'], hidden_dim = 512)

            self.R_Z_inputs = False
            self.phi_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1,1], [1]]


        else:
            RZ_shape = config['SHAPE_FINAL'][1:]

            self.R_Z_inputs = config.get('R_Z_INPUT', False)
            self.phi_inputs = config.get('PHI_INPUT', False)

            in_channels = 1

            dataset_num = config['DATASET_NUM']
            self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = RZ_shape, dataset_num = dataset_num)
            self.phi_image = create_phi_image(device, shape = RZ_shape, dataset_num = dataset_num)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [[1,cond_size]], [1]]


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), cond_size = cond_size, time_embed = (self.time_embed == 'sin')  )

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


    def set_sampling_steps(self, nsteps):
        self.nsteps = nsteps
        #precompute useful quantities for sampling
        self.betas = cosine_beta_schedule(self.nsteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

        #shift all elements over by inserting unit value in first place
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

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
        return torch.squeeze(self.avg_showers[idxs]), torch.squeeze(self.std_showers[idxs])


    #utils function for min snr weighting
    def get_scalings(self, sigma, sigma_data = 1.0):
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def weighting_soft_min_snr(self, sigma, k = 2, sigma_data = 1.0):
        return (sigma * sigma_data) ** 2 / (sigma ** 2 + sigma_data ** k) ** 2




    def compute_loss(self, data, E, model = None, noise = None, t = None, layers = None, loss_type = "l2", rnd_normal = None, layer_loss = False, scale=1 ):
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

        if (model is None):
            model = self.model if not layer_loss else self.layer_model



        x0_pred = self.denoise(x_noisy, E=E, sigma=sigma, model = model, layers = layers)


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
            
            
                
        if('minsnr' in self.training_obj):
            snr_weight = self.weighting_soft_min_snr(sigma)
            weight *= snr_weight


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

    def pred(self, x, E, t_emb, model = None, layers = None, layer_sample = False, controls = None):
        if(model is None): model = self.model


        if(self.NN_embed is not None and not layer_sample): x = self.NN_embed.enc(x).to(x.device)
        if(self.layer_cond and layers is not None): E = torch.cat([E, layers], dim = 1)
        out = model(self.add_RZPhi(x), cond=E, time=t_emb, controls = controls)
        if(self.NN_embed is not None and not layer_sample): out = self.NN_embed.dec(out).to(x.device)
        return out

    def denoise(self, x, E =None, sigma=None, model = None, layers = None, layer_sample = False, controls = None):
        t_emb = self.do_time_embed(embed_type = self.time_embed, sigma = sigma.reshape(-1))
        #if('minsnr' in self.training_obj): sigma_data = 0.5
        #else: sigma_data = 1.0
        c_skip, c_out, c_in = self.get_scalings(sigma)


        pred = self.pred(x * c_in, E, t_emb, model = model, layers = layers, layer_sample = layer_sample, controls = controls)

        if('noise_pred' in self.training_obj):
            return (x - sigma * pred)

        elif('mean_pred' in self.training_obj):
            return pred
        elif('hybrid' in self.training_obj):
            return (c_skip * x + c_out * pred)
        else:
            print("??? Training obj %s" % self.training_obj)


    def __call__(self, x, **kwargs):
        #sometimes want to call Unet directly, sometimes need wrappers
        if('cond' in kwargs.keys()):
            return self.model(x, **kwargs)
        else:
            return self.denoise(x, **kwargs)




    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)
        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)
        cold_scales = cold_noise_scale
        return torch.add(avg_shower, cold_scales * (noise * std_shower))

    def dpm_sampler(self, x_start, E, layers = None, num_steps = 400, sample_algo = 'dpm', model = None, debug = False, extra_args = None):
        #dpm family of samplers

        old_nsteps = self.nsteps
        if(self.nsteps != num_steps):
            self.set_sampling_steps(num_steps)

        gen_size = E.shape[0]
        device = x_start.device 

        xs = []
        x0s = []

        time_steps = list(range(0, num_steps))
        time_steps.reverse()

        #scale starting point to appropriate noise level
        sigmas = torch.tensor([self.sqrt_one_minus_alphas_cumprod[num_steps - t -1]/ self.sqrt_alphas_cumprod[num_steps - t -1] for t in torch.arange(num_steps)])
        sigma_min = sigmas[-1]
        sigma_max = sigmas[0]

        x = x_start * sigmas[0]
        if('adapt' in sample_algo):
            x = sample_dpm_adaptive(self, x, sigma_min, sigma_max, extra_args={'E':E, 'layers':layers})
        elif('++' in sample_algo and 'sde' in sample_algo):
            x = sample_dpmpp_2m_sde(self, x, sigmas, extra_args={'E':E, 'layers':layers})
        elif('++' in sample_algo):
            x = sample_dpmpp_2m(self, x, sigmas, extra_args={'E':E, 'layers':layers})
        elif('unipc' in sample_algo):
            x = sample_unipc(self, x, sigmas, extra_args={'E':E, 'layers':layers})
        else:
            x = sample_dpm_fast(self, x, sigma_min, sigma_max, num_steps, extra_args={'E':E, 'layers':layers})

        return x, None,None



    @torch.no_grad()
    def Sample(self, E, layers = None, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, model = None, gen_shape = None, layer_sample = False):
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
        x_start = torch.randn(gen_shape, device=device, dtype=torch.float32)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)

        caller = self if (model is None or layer_sample) else model
        extra_args = {'E':E, 'layers':layers, 'layer_sample' : layer_sample, 'model' : model}

        if('euler' in sample_algo or 'edm' in sample_algo or 'heun' in sample_algo or 'dpm2' in sample_algo or 'restart' in sample_algo or 'lms' in sample_algo):
            S_churn = 40 if (sample_algo == 'edm' or 'noise' in sample_algo) else 0  #Number of steps to 'reverse' to add back noise
            S_min = 0.01
            S_max = 50 if (sample_algo == 'edm' or 'noise' in sample_algo) else 1
            S_noise = 1.003
            sigma_min = 0.002
            #sigma_max = 500.0
            sigma_max = 80.0
            orig_schedule = False

            x,xs, x0s = edm_sampler(model,x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max, 
                    S_churn = S_churn, S_min = S_min, S_max = S_max, S_noise = S_noise, sample_offset = sample_offset, orig_schedule = orig_schedule, extra_args=extra_args)

        elif('consis' in sample_algo):
            orig_num_steps = self.nsteps

            self.set_sampling_steps(self.consis_nsteps)
            print("Consis steps %i" % self.consis_nsteps)

            #hardcoded for now...
            sample_idxs = [0, int(round(self.consis_nsteps*0.55)), int(round(self.consis_nsteps*0.75)), int(round(self.consis_nsteps*0.90)), int(round(self.consis_nsteps*0.95))]

            t_all_steps = [self.sqrt_one_minus_alphas_cumprod[self.consis_nsteps - t -1]/ self.sqrt_alphas_cumprod[self.consis_nsteps - t -1] for t in torch.arange(self.consis_nsteps)]

            if(num_steps > 1):
                t_steps = torch.tensor([t_all_steps[i] for i in sample_idxs[:num_steps]])
            else:
                t_steps = torch.tensor([t_all_steps[0]])
            sigmas = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # end point is zero noise

            x,xs, x0s = sample_consis(caller, x_start, sigmas, extra_args = extra_args)
            self.set_sampling_steps(orig_num_steps)

        elif(sample_algo == 'ddim' or sample_algo =='ddpm'):
            x, xs, x0s = sample_dd(caller, x_start, num_steps, sample_offset = sample_offset, sample_algo = sample_algo, debug = debug, extra_args=extra_args )

        elif('dpm' in sample_algo):
            x, xs, x0s = self.dpm_sampler(x_start, E, layers = layers, num_steps = num_steps, sample_algo = sample_algo, debug = debug, model = model, extra_args = extra_args)
        else:
            print("Unrecognized sampling algo %s" % sample_algo)
            exit(1)

        if(debug):
            return x.detach().cpu().numpy(), xs, x0s
        else:   
            return x.detach().cpu().numpy()


    
        
