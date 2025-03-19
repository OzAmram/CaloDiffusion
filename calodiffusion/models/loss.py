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

        losses = {
            "l1": lambda y_hat, y, weight=1: torch.nn.functional.l1_loss(y_hat, y),
            "l2": l2_loss,
            "mse": lambda y_hat, y, weight=1: torch.nn.functional.mse_loss(y_hat, y),
            "huber": lambda y_hat, y, weight=1: torch.nn.functional.smooth_l1_loss(y_hat, y),
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
