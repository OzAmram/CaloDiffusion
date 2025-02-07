from typing import Optional
import copy 
import torch 

from calodiffusion.models.calodiffusion import CaloDiffusion
from calodiffusion.models.models import ResNet
from calodiffusion.utils import utils

class LayerDiffusion(CaloDiffusion):
    def __init__(self, config, n_steps = 400, loss_type = 'l2'):
        super().__init__(config, n_steps, loss_type)
        self.layer_loss = False 
        sampler_algo = self.config.get("LAYER_SAMPLER", "DDim")
        self.layer_sampler = utils.load_attr("sampler", sampler_algo)(self.config, sampler_algo.lower())
        self.layer_steps = self.config.get("LAYER_STEPS", n_steps)
    
    def init_model(self):
        self.layer_model = ResNet(dim_in = self.config['SHAPE_PAD'][2] + 1, num_layers = 5).to(device = self.device)
        model = super().init_model()
        self.base_model = model
        return model

    def load_state_dict(self, state_dict, strict = True):
        layer_model_state_dict = torch.load(self.config['layer_model'], map_location=self.device, weights_only=False)
        self.layer_model.load_state_dict(layer_model_state_dict['model_state_dict'], strict)
        
        base_model_state_dict = {".".join(key.split(".")[1:]): state for key, state in state_dict.copy().items() if "NN_embed" not in key}
        self.base_model.load_state_dict(base_model_state_dict, strict)
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['layer_model'] = self.layer_model.state_dict()
        return state_dict

    def denoise(self, x, E=None, sigma=None, layers = None):
        if self.layer_loss: 
            self.model = self.layer_model
            
        x_0 = super().denoise(x, E, sigma, layers)

        self.model = self.base_model
        return x_0
    
    def sample_layers(self, energy, layers, debug = False, sample_offset = None):
        self.model = self.layer_model
        self.layer_loss = True

        shape = (energy.shape[0], self.config['SHAPE_PAD'][2]+1)
        start = self.noise_generation(shape).to(torch.float32)

        x, _, _ = self.layer_sampler(
            self,
            start, 
            energy, 
            layers,
            self.layer_steps,
            sample_offset,
            debug
        )
        self.layer_loss = False
        self.model = self.base_model
        return x.detach().cpu().numpy()

    def sample(
        self,
        energy: torch.Tensor,
        layers: list,
        num_steps: int = 400,
        debug: bool = False,
        sample_offset: Optional[int] = None,
    ) -> torch.Tensor:
        
        generated_shape = list(copy.copy(self._data_shape))
        generated_shape.insert(0,  energy.shape[0])

        start = self.noise_generation(generated_shape).to(torch.float32)
        layers = self.sample_layers(energy, layers, debug, sample_offset)

        x, xs, x0s = self.sampler_algorithm(
            self,
            start, 
            energy, 
            layers,
            num_steps,
            sample_offset,
            debug
        )

        if debug:
            return x.detach().cpu().numpy(), xs, x0s
        else:
            return x.detach().cpu().numpy()
