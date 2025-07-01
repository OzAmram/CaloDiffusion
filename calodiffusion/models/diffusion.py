"""
Generic class subclassing torch.nn for running a diffusion model

Includes sampling for generation, predictive forward, loss calculation
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
from typing import Optional, Union
import numpy as np
import torch

from calodiffusion.utils import utils
from calodiffusion.utils import plots as plot


class Diffusion(torch.nn.Module, ABC):
    def __init__(self, config: Union[str, dict], n_steps: int = 400, loss_type: str = 'l2'):
        super().__init__()

        self.config = config if isinstance(config, dict) else utils.LoadJson(config)
        self.device = utils.get_device()
        self.tqdm = utils.import_tqdm()

        self.nsteps = n_steps
        self.loss_type = loss_type

        self.hgcal = self.pre_embed = False

        loss_algo = self.config.get('TRAINING_OBJ', "noise_pred")
        self.loss_function = utils.load_attr("loss", loss_algo)(self.config, self.nsteps, self.loss_type)

        sampler_algo = self.config.get("SAMPLER", "DDim")
        self.sampler_algorithm = utils.load_attr("sampler", sampler_algo)(self.config)

        self.NN_embed = None
        #self.NN_embed = self.init_embedding_model()

        if "orig" not in self.config.get("SHOWER_EMBED", ""): 
            self._data_shape =  self.config["SHAPE_PAD"][1:]
        else: 
            self._data_shape = self.config["SHAPE_ORIG"][1:]

    @abstractmethod
    def init_model(self): 
        """
        Create the forward model
        """
        raise NotImplementedError

    def init_embedding_model(self): 
        """
        Initialize and load the embedding model
        """
        return None

    @abstractmethod
    def noise_generation(self, shape): 
        # Totally fine to use super, but it should be explicitly specified. 
        return torch.randn(shape, device=self.device, dtype=torch.float32)

    @abstractmethod
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function that makes predictions that interaction with the loss function
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x_noisy, E, sigma, model, layers):
        """
        Denoise the model output
        """
        raise NotImplementedError

    def sample(
        self,
        energy: torch.Tensor,
        layers: list,
        num_steps: int = 400,
        debug: bool = False,
        sample_offset: Optional[int] = 0,
    ) -> torch.Tensor:
        
        generated_shape = list(copy.copy(self._data_shape))
        generated_shape.insert(0,  energy.shape[0])

        start = self.noise_generation(generated_shape)

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
        
    def compute_loss(self, data, energy, noise, layers, time=None, rnd_normal=None):
        """
        Compute loss for a single model step
        """
        return self.loss_function(self, data, energy, noise=noise, layers=layers, rnd_normal=rnd_normal)

    def load_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True
    ):
        return super().load_state_dict(state_dict, strict)


    def generate(
        self,
        data_loader: utils.DataLoader,
        sample_steps: int,
        debug: bool = False,
        sample_offset: Optional[int] = 0,
        sparse_decoding: Optional[bool] = False,
    ):
        """
        Generate samples for a whole dataloader
        """
        shower_embed = self.config.get("SHOWER_EMBED", "")
        orig_shape = "orig" in shower_embed

        generated = []
        data = []
        energies = []
        layers = []

        for E, layers_, d_batch in self.tqdm(data_loader):
            E = E.to(device=self.device)
            d_batch = d_batch.to(device=self.device)
            layers_ = layers_.to(device=self.device)

            batch_generated = self.sample(
                E,
                layers=layers_,
                num_steps=sample_steps,
                debug=debug,
                sample_offset=sample_offset,
            )

            if debug:
                data.append(d_batch.detach().cpu().numpy())

            E = E.detach().cpu().numpy()
            energies.append(E)
            if "layer" in self.config["SHOWERMAP"]:
                layers.append(layers_.detach().cpu().numpy())

            # Plot the histograms of normalized voxels for both the diffusion model and Geant4
            if debug:
                gen = self._debug_sample_plot(batch_generated, data)
                generated.append(gen)
            else: 
                generated.append(batch_generated)

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
            sparse_decoding=sparse_decoding,
        )
        if not orig_shape:
            generated = generated.reshape(self.config["SHAPE_ORIG"])

        energies = np.reshape(energies,(energies.shape[0],-1))

        return generated, energies

    def _debug_sample_plot(self, batch_inference, batch_data, plot_folder:str="./plots/"): 
        generated, all_gen, x0s = batch_inference
        for j in [
            0,
            len(all_gen) // 4,
            len(all_gen) // 2,
            3 * len(all_gen) // 4,
            9 * len(all_gen) // 10,
            len(all_gen) - 10,
            len(all_gen) - 5,
            len(all_gen) - 1,
        ]:
            fout_ex = f"{plot_folder}/{self.config['CHECKPOINT_NAME']}_{self.__name__}_norm_voxels_gen_step{j}.png"
            plot.ScatterESplit(flags=None, config=self.config)._hist(
                [
                    all_gen[j].cpu().reshape(-1),
                    np.concatenate(batch_data).reshape(-1),
                ],
                ["Diffu", "Geant4"],
                ["blue", "black"],
                xaxis_label="Normalized Voxel Energy",
                num_bins=40,
                normalize=True,
                fname=fout_ex,
            )

            fout_ex = f"{plot_folder}/{self.config['CHECKPOINT_NAME']}_{self.__name__}_norm_voxels_x0_step{j}.png"
            plot.ScatterESplit(flags=None, config=self.config)._hist(
                [x0s[j].cpu().reshape(-1), np.concatenate(batch_data).reshape(-1)],
                ["Diffu", "Geant4"],
                ["blue", "black"],
                xaxis_label="Normalized Voxel Energy",
                num_bins=40,
                normalize=True,
                fname=fout_ex,
            )
        return generated
