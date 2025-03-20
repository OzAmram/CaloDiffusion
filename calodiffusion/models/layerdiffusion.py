from typing import Literal, Optional
import copy 
import numpy as np
import torch 

from calodiffusion.models.calodiffusion import CaloDiffusion
from calodiffusion.models.models import ResNet
from calodiffusion.utils import utils

class LayerDiffusion(CaloDiffusion):
    """
    Creates a model which has two steps, a layer model and a base model. 
    The layer model takes the existing layers and encodes them in the same shape as the original layer, but as a pre-training step. 
    Then during inference, a trained mode (base_model) uses those layer encodings to generate the final output.

    During training, only the layer model is used. Both are used during inference. 
    """

    def __init__(self, config, n_steps = 400, loss_type = 'l2'):
        super().__init__(config, n_steps, loss_type)
        self.layer_loss = False 
        sampler_algo = self.config.get("LAYER_SAMPLER", "DDim")
        self.layer_sampler = utils.load_attr("sampler", sampler_algo)(self.config)
        self.layer_steps = self.config.get("LAYER_STEPS", n_steps)
        self.base_forward = self.forward

        self.shape_pad = self.config.get("SHAPE_PAD")
        if self.shape_pad is None: 
            self.shape_pad = self.config['SHAPE_FINAL']

    def init_model(self):
        cond_size = 3 if self.hgcal else 1
        self.layer_model = ResNet(dim_in = self.config['SHAPE_FINAL'][2] + 1, num_layers = 5, cond_size = cond_size).to(device = self.device)
        model = super().init_model().to(device = self.device)
        self.base_model = model
        return model

    def set_layer_state(self, is_layer=False): 
        if is_layer: 
            self.layer_loss = True
            self.model = self.layer_model
            self.forward = self.layer_forward
        else: 
            self.layer_loss = False
            self.model = self.base_model
            self.forward = self.base_forward

    def compute_loss(self, data, energy, noise, layers, time=None, rnd_normal=None):
        if self.layer_loss: 
            noise = self.noise_generation(layers.shape).to(torch.float32)
            return super().compute_loss(layers.to(torch.float32), energy, noise, layers, time, rnd_normal)
        else: 
            return super().compute_loss(data, energy, noise, layers, time, rnd_normal)

    def load_layer_model_state(self, strict = True): 
        layer_model_state_dict = torch.load(self.config['layer_model'], map_location=self.device, weights_only=False)
        state_dict = layer_model_state_dict if 'model_state_dict' not in layer_model_state_dict else layer_model_state_dict['model_state_dict']
        
        weights_prefixes = set([key.split('.')[0] for key in state_dict.keys()])
        if "layer_model" in weights_prefixes: 
            state_dict = {key.removeprefix("layer_model."): value for key, value in state_dict.items()}

        try: 
            self.layer_model.load_state_dict(state_dict=state_dict, strict=strict)
        except RuntimeError as e: 
            if ("size mismatch" in str(e)) or ("Missing key(s) in state_dict" in str(e)): 
                raise RuntimeError(e)
            else: 
                self.layer_model.load_state_dict(state_dict=state_dict, strict=False)

    def load_state_dict(self, state_dict, strict = True):
        self.load_layer_model_state(strict)        
        # Load the base model 
        
        weights_prefixes = set([key.split('.')[0] for key in state_dict.keys()])
        if "base_model" in weights_prefixes: 
            state_dict = {key.removeprefix("base_model."): value for key, value in state_dict.items()}

        elif "model" in weights_prefixes: 
            state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}

        try: 
            self.base_model.load_state_dict(state_dict, strict)
        except RuntimeError as e: 
            if ("size mismatch" in str(e)) or ("Missing key(s) in state_dict" in str(e)): 
                raise RuntimeError(e)
            else: 
                self.base_model.load_state_dict(state_dict, strict=False)
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['layer_model'] = self.layer_model.state_dict()
        return state_dict
    
    def layer_forward(self, x, E, time, **kwargs):
        rz_phi = self.add_RZPhi(x)
        out = self.model(rz_phi, cond=E.to(torch.float32), time=time.to(torch.float32), controls=kwargs)
        return out
    
    def sample_layers(self, energy, layers, debug = False, sample_offset = None):
        self.set_layer_state(is_layer=True)
        shape = (energy.shape[0], self.shape_pad[2]+1)
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
        self.set_layer_state(is_layer=False)
        return x

    def sample(
        self,
        energy: torch.Tensor,
        layers: list,
        num_steps: int = 400,
        debug: bool = False,
        sample_offset: Optional[int] = None,
        return_layers: bool = False
    ) -> dict[Literal['x', 'x0s', 'xs', 'layers'], torch.tensor]:
        
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
        out = {"x": x.detach().cpu().numpy()}
        if debug: 
            out['xs'] = xs
            out['x0s'] = x0s
        
        if return_layers: 
            out['layers'] = layers

        return out

    def generate(
        self,
        data_loader: utils.DataLoader,
        sample_steps: int,
        debug: bool = False,
        sample_offset: Optional[int] = 0,
    ):
        """
        Generate samples for a whole dataloader

        Layer-CaloDiffusion assumes no layers are given - all generated by the layer_model
        """
        shower_embed = self.config.get("SHOWER_EMBED", "")
        orig_shape = "orig" in shower_embed

        generated = []
        data = []
        energies = []
        layers = []

        for E, _, d_batch in self.tqdm(data_loader):
            E = E.to(device=self.device)
            d_batch = d_batch.to(device=self.device)

            batch_generated = self.sample(
                E,
                layers=None,
                num_steps=sample_steps,
                debug=debug,
                sample_offset=sample_offset,
                return_layers=True
            )
            
            layers_ = batch_generated['layers']

            if debug:
                data.append(d_batch.detach().cpu().numpy())

            E = E.detach().cpu().numpy()
            energies.append(E)
            layers.append(layers_.detach().cpu().numpy())

            # Plot the histograms of normalized voxels for both the diffusion model and Geant4
            if debug:
                gen = self._debug_sample_plot(batch_generated, data) # TODO correct batch_generated into the tuple of 'x, xs, x0s'
                generated.append(gen)
            else: 
                generated.append(batch_generated['x'])

        generated = np.concatenate(generated)
        energies = np.concatenate(energies)
        layers = np.concatenate(layers)

        generated, energies = utils.ReverseNorm(
            generated,
            energies,
            shape=self.config["SHAPE_FINAL"],
            config = self.config,
            emax=self.config["EMAX"],
            emin=self.config["EMIN"],
            layerE=layers,
            logE=self.config["logE"],
            binning_file=self.config["BIN_FILE"],
            max_deposit=self.config["MAXDEP"],
            showerMap=self.config["SHOWERMAP"],
            dataset_num=self.config.get("DATASET_NUM", 2),
            orig_shape=orig_shape,
            ecut=float(self.config["ECUT"]),
            hgcal=self.hgcal,
            embed=self.pre_embed,
            NN_embed=self.NN_embed,
        )
        if not orig_shape:
            generated = generated.reshape(self.config["SHAPE_ORIG"])

        energies = np.reshape(energies,(energies.shape[0],-1))

        return generated, energies