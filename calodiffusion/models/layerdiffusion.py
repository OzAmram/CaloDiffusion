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
        self.layer_sampler = utils.load_attr("sampler", sampler_algo)(self.config)
        self.layer_steps = self.config.get("LAYER_STEPS", n_steps)
        self.base_forward = self.forward

    def init_model(self):
        self.layer_model = ResNet(dim_in = self.config['SHAPE_PAD'][2] + 1, num_layers = 5).to(device = self.device)
        model = super().init_model().to(device = self.device)
        self.base_model = model
        return model

    def load_state_dict(self, state_dict, strict = True):
        layer_model_state_dict = torch.load(self.config['layer_model'], map_location=self.device, weights_only=False)
        layer_model_state = {".".join(key.split(".")[1:]): state for key, state in layer_model_state_dict['model_state_dict'].copy().items() if "layer_model" in key}
        layer_model_state.pop('', None)  # Can have a leftover "" in the keys. Strange. 
        self.layer_model.load_state_dict(layer_model_state, strict)
        
        base_model_state_dict = {".".join(key.split(".")[1:]): state for key, state in state_dict.copy().items() if "NN_embed" not in key or "layer_model" not in key}
        self.base_model.load_state_dict(base_model_state_dict, strict=False)
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['layer_model'] = self.layer_model.state_dict()
        return state_dict
    
    def layer_forward(self, x, E, time, **kwargs):
        rz_phi = self.add_RZPhi(x)
        out = self.model(rz_phi, cond=E.to(torch.float32), time=time.to(torch.float32), controls=kwargs)
        return out
    
    def sample_layers(self, energy, layers, debug = False, sample_offset = None):
        self.model = self.layer_model
        self.layer_loss = True
        self.forward = self.layer_forward
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
        self.forward = self.base_forward
        return x

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
        layers = self.sample_layers(energy, layers=None, debug=debug, sample_offset=sample_offset)

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
