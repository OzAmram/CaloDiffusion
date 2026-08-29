from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch 

from calodiffusion.utils import utils, sampling

# TODO Fix class names to be in camel-case

class Loss(ABC): 
    def __init__(self, config, n_steps, loss_type='l1') -> None:
        self.config = config

        self.update_step(n_steps)
        self.discrete_time = True
        self.P_mean = -1
        self.P_std = 1
        self.sigma_data = 0.5
        if "log" in config.get("NOISE_SCHED", "linear"):
            self.discrete_time = False

            # TODO Pull from config
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 1.0

        self.loss = self._loss(loss_type=loss_type)

    def get_scaling(self, sigma): 
        """
            Scale input to the forward model before prediction, as required by the loss function. 
        """
        out = {
                'c_skip': self.sigma_data**2 / (sigma**2 + self.sigma_data**2),
                'c_out': sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5,
                'c_in' : 1 / (sigma**2 + self.sigma_data**2) ** 0.5,
            }



        return out

    @abstractmethod
    def loss_function(self, model, data, E, time, sigma=None, noise=None, layers=None ): 
        """
            Compute the loss for a model - pass data and E forward, produing a single step

        Args:
            model (_type_): Model with a forward function (child of 'Diffusion' class.)
            data (_type_): Batch of data to operation on
            E (_type_): Energy of the batch
            sigma (_type_, optional): Data sigma. Defaults to None.
            noise (_type_, optional): Noise to be applied to the batch before prediction. Defaults to None.
            layers (_type_, optional): Additional layers to compute prediction for. Requires that the model is a layer model. Defaults to None.
            layers (_type_, optional): Use predefined noise levels 

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def apply_scaling_skips(self, prediction, x, c_in, c_skip, c_out, sigma=None): 
        """
        Scaling applied to prediction after the model forward, requires skip connection calculation. 

        Args:
            prediction (_type_): _description_
            x (_type_): _description_
            c_in (_type_): _description_
            c_skip (_type_): _description_
            c_out (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def update_step(self, steps:int): 
        self.n_steps = steps

        betas = sampling.cosine_beta_schedule(self.n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # shift all elements over by inserting unit value in first place
        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _loss(self, loss_type: str) -> callable: 
        """
        Return the loss function
        Each loss function takes the target, predictions, and weights for each prediction (weight is only used for l2.)
        """

        def l2_loss(target, prediction, weight): 
            return  (weight * ((prediction - target) ** 2)).sum() / (torch.mean(weight) * np.prod(target.shape))

        def l2_decay(target, prediction, weight):
            flatten_dim = 3  ## Flatten across the feature dimension 
            loss = (torch.clamp_min(((x - y) ** 2).flatten(flatten_dim).sum(-1), 1e-8)).sqrt() / (np.prod(y.shape[flatten_dim:])) / self.sigma

        losses = {
            "l1": lambda y_hat, y, weight=1: torch.nn.functional.l1_loss(y_hat, y),
            "l2": l2_loss,
            "mse": lambda y_hat, y, weight=1: torch.nn.functional.mse_loss(y_hat, y),
            "huber": lambda y_hat, y, weight=1: torch.nn.functional.smooth_l1_loss(y_hat, y),
            "l2_decay": l2_decay
        }

        if loss_type not in losses.keys(): 
            raise NotImplementedError("Loss type %s not implemented, pick from (%s)", loss_type, losses.keys())
        
        return losses[loss_type]

    def __call__(self, model, data, E, noise=None, time=None, layers=None, rnd_normal=None) -> Any:

        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if noise is None:
            noise = torch.randn_like(data)

        if self.discrete_time:
            if time is None:
                time = torch.randint(
                    0, self.n_steps, (data.size()[0],), device=data.device
                ).long()

            sqrt_alphas_cumprod_t = utils.subsample_alphas(
                self.sqrt_alphas_cumprod, time, data.shape
            )
            sqrt_one_minus_alphas_cumprod_t = utils.subsample_alphas(
                self.sqrt_one_minus_alphas_cumprod, time, data.shape
            )
            sigma = sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t

        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp().reshape(const_shape)

        return self.loss_function(model, data, E, sigma=sigma, noise=noise, layers=layers)

class minsnr(Loss): 
    def __init__(self, config, n_steps) -> None:
        # From https://arxiv.org/pdf/2303.09556 
        super().__init__(config, n_steps)

    def loss_function(self, model, data, E, sigma=None, noise=None, layers=None):
        # TODO TO CHECK...
        # Sigma0 or sigma?
        x_noisy = data + sigma * noise
        sigma0 = (noise * self.P_std + self.P_mean).exp()
        scales = self.get_scaling(sigma)
        pred = model.denoise(x_noisy * scales['c_in'], E, sigma, layers=layers)
        pred = data - sigma * pred

        target = (data - scales['c_skip'] * x_noisy) / scales['c_out']

        weight = torch.ones_like(pred)
        return self.loss(pred, target, weight)

class hybrid_weight(Loss): 
    def __init__(self, config, n_steps, loss_type='l1') -> None:
        super().__init__(config, n_steps, loss_type)

    
    def loss_function(self, model, data, E, sigma=None, noise=None, layers=None):
        x_noisy = data + sigma * noise
        x0_pred = model.denoise(
                x_noisy, E=E, sigma=sigma, layers=layers
            )
        
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        weight = torch.reshape(1.0 + (1.0 / sigma**2), const_shape)
        target = data
        pred = x0_pred

        return self.loss(pred, target, weight)

class noise_pred(Loss):
    def __init__(self, config, n_steps, loss_type='l1') -> None:
        super().__init__(config, n_steps, loss_type)

    def loss_function(self, model, data, E, sigma=None, noise=None, layers=None):
        x_noisy = data + sigma * noise
        x0_pred = model.denoise(
                x_noisy, E=E, sigma=sigma, layers=layers
            )
        x0_pred = data - sigma * x0_pred
        pred = (data - x0_pred) / sigma
        target = noise

        weight = torch.ones_like(pred)
        return self.loss(pred, target, weight)
    
class mean_pred(Loss): 
    def __init__(self, config, n_steps, loss_type='l1') -> None:
        super().__init__(config, n_steps, loss_type)

    def loss_function(self, model, data, E, sigma=None, noise=None, layers=None):
        x_noisy = data + sigma * noise
        x0_pred = model.denoise(
                x_noisy, E=E, sigma=sigma, layers=layers
            )
        target = data
        weight = 1.0 / (sigma**2)
        pred = x0_pred

        return self.loss(pred, target, weight)


class IMM(Loss): 
    def __init__(self, config, n_steps, loss_type='l1') -> None:
        """
        A kernel based MMD loss function, as described in https://arxiv.org/abs/2503.07565
        L = 1/2 * (K(f_t, f_t) + K(f_r, f_r) - 2 * K(f_t, f_r))
        where K is a kernel (self.kernel), and f_t, f_r are the embeddings of the high and low noise samples, respectively.
        See sampling.PushforwardTraining for more details on how the samples are generated.

        """
        super().__init__(config, n_steps, loss_type)
        pad_shape = config.get("SHAPE_PAD", [-1, 1, 28, 12, 21])

        self.feature_dimension = pad_shape.index(max(pad_shape[1:]))
        self.bandwidth = config.get("IMM_BANDWIDTH", 1.0)
        self.loss_cutoff = config.get("IMM_LOSS_CUTOFF", 1e-8)

        # Pushfoward sampler used only during training 
        self.sampler = utils.load_attr("sampler", "PushforwardTraining")(self.config)

    def kernel(self, f_a, f_b, weight=1.0): 
        """
        Compute the kernel for IMM Loss - exp(-||x - y||₂ / ||x|| * w)  
        f_a, f_b are two embeddings, which can be the same. 
        """
        # Compute the L2 distance, normalized by the feature dimension
        # Take the l2 norm, clamp, and normalize with bandwidth and feature dimensions
        batch_size = f_a.shape[0] # cdist assumes a tensor with shape (N, batch, features)
        f_a = f_a.reshape(batch_size, -1)
        f_b = f_b.reshape(batch_size, -1)
        weight = weight.reshape(batch_size, -1)

        clamped_l2 = torch.clamp_min(torch.cdist(f_a, f_b, p=2), self.loss_cutoff) / (f_a.shape[-1] * self.bandwidth)
        # Apply RBF transformation: exp(-distance * weight)


        embedding_similarity = torch.exp(-clamped_l2 * weight)
        # Remove the diagonal elements of the if f_a and f_b are the same
        # if torch.allclose(f_a, f_b, atol=1e-10): 
        #     embedding_similarity = embedding_similarity[~torch.eye(batch_size, dtype=bool, device=f_a.device)]

        return torch.mean(embedding_similarity)
    
    def _calculate_weight(self, t, s):
        """
        Weight for the kernel -> adaptive based on time difference
        w = 1/c_out, c_out(t, s) is the scaling factor from Simple-EDM
        """
        sigma_t, alpha_t = self.sampler.get_alpha_sigma(t)
        # Clamp the weight to make sure nothing explodes

        intermediate = torch.clamp_min(torch.abs(sigma_t * self.sigma_data), self.loss_cutoff)
        weight = torch.abs(torch.sqrt(alpha_t**2 + sigma_t**2)) / intermediate

        return torch.clamp(weight, min=0.1, max=10.0)

    def loss_function(self, model, data, E, sigma=None, noise=None, layers=None):
        """An MMD loss between two points of sampling using pushforward to access the two points"""

        if self.feature_dimension >= data.ndim:
            self.feature_dimension = data.ndim - 1

        sampled_high_noise, sampled_low_noise, time_t, time_s = self.sampler(
            model=model, 
            sigma=sigma,
            x=data, 
            energy=E, 
            layers=layers)

        weight = self._calculate_weight(time_t, time_s)
        # Unsqueeze the sampled tensors 
        high_noise_intersample = self.kernel(sampled_high_noise.unsqueeze(1), sampled_high_noise.unsqueeze(0), weight=weight)
        low_noise_intersample = self.kernel(sampled_low_noise.unsqueeze(1), sampled_low_noise.unsqueeze(0), weight=weight)
        cross_sample = self.kernel(sampled_high_noise.unsqueeze(1), sampled_low_noise.unsqueeze(0), weight=weight)
        return high_noise_intersample + low_noise_intersample - (2 * cross_sample)
