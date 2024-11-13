from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
import optuna
import jetnet

from datetime import datetime
from calodiffusion.train import Train
from calodiffusion.utils import utils
import calodiffusion.utils.HighLevelFeatures as HLF

class Optimize: 
    def __init__(self, flags, trainer: type[Train], objectives: list[Literal["COUNT", "FPD"]]) -> None:

        implemented_objectives: dict[str, type[Objective]] = {
            "COUNT": Count(), 
            "FPD": FPD()
        }
        self.flags = flags
        self.trainer = trainer
        self.objectives = []
        for objective in objectives: 
            self.objectives.append(implemented_objectives[objective])

    def train(self, trial_config): 
        config = self.suggest_config(trial_config)
        train_model = self.trainer(flags=self.flags, config=config, save_model=False)
        model, _, _ = train_model.train()
        eval_data = train_model.loader_val
        return model, eval_data, config

    def suggest_config(self, trial_config): 
        config = utils.LoadJson(self.flags.config)
        optimized_section = config.get("OPTIMIZE", {})
        for key, values in optimized_section.items(): 
            if not isinstance(values, Iterable): 
                raise ValueError("All optimization parameters must be given as a list.")

            if key == "LAYER_SIZE_UNET": 
                init_size = trial_config.suggest_int("init_unet", *values["init_unet"], step=2) # Always an even number
                n_layers = trial_config.suggest_int("n_unet_layers", *values["n_unet_layers"])
                final_layer = int(trial_config.suggest_int("layer_ratio", *values['layer_ratio']) * init_size)
                
                unet_layers = [init_size for _ in range(n_layers)]
                unet_layers.append(final_layer)
                config[key] = unet_layers
                config['BLOCK_GROUPS'] = int(init_size/2) # TODO see how flexible this is. 

            # This is a GROSS way to do this
            else: 
                if all([isinstance(i, str) for i in values]) or (True in values): 
                    config[key] = trial_config.suggest_categorical(key, values)
                elif all([isinstance(i, int) for i in values]): 
                    config[key] = trial_config.suggest_int(key, *values)       
                else: 
                    config[key] = trial_config.suggest_float(key, *values)
        return config

    def suggest_sampler_config(self, trial_config): 
        ""

    def eval(self, model, eval_data, config) -> Sequence: 
        return [obj(model, eval_data, config) for obj in self.objectives]

    def objective(self, trial) -> tuple: 
        try: 
            model, eval_data, config = self.train(trial)
        except RuntimeError as err:
            if "Kernel size can't be greater than actual input size" in str(err): 
                objectives = [obj.failure() for obj in self.objectives]
                return objectives
            else: 
                raise RuntimeError(err)
            
        objectives = self.eval(model, eval_data, config)
        return objectives

    def __call__(self) -> None:
        study = optuna.create_study(
            study_name=self.flags.study_name, 
            load_if_exists=True, 
            directions=[obj.direction() for obj in self.objectives]
            )
        study.optimize(
            self.objective, 
            n_trials=30, 
            timeout=300
        )
        

class Objective(ABC): 
    @staticmethod
    @abstractmethod
    def direction() -> Literal['minimize', "maximize"]: 
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def failure() -> float: 
        "What is returned if the model has failed to train"
        raise NotImplementedError
    
    @staticmethod
    def __call__(trained_model, eval_data, kwargs) -> float:
        raise NotImplementedError
    

class Count(Objective): 
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "minimize"
    
    @staticmethod
    def failure():
        return 10e8

    @staticmethod
    def get_forward(): 
        class ModelForward(torch.nn.Module): 
            def __init__(self, model, eval_data, sample_steps, sample_offset) -> None:
                super().__init__()
                self.model = model
                self.eval_data = eval_data
                self.E, self.layers, _ = next(iter(eval_data))
                self.sample_steps = sample_steps
                self.sample_offset = sample_offset

            def __call__(self, x=None) -> Any:
                self.model.generate(
                    data_loader=self.eval_data, 
                    sample_steps=self.sample_steps, 
                    sample_offset=self.sample_offset
                )
                
        return ModelForward

    @staticmethod
    def __call__(trained_model, eval_data, trial_config) -> float:
        random = np.random.default_rng()
        weight_matrix = random.random((24, 24))
        weight_matrix_compare = random.random((24, 24))

        forward = Count.get_forward()(
            model=trained_model, 
            eval_data=eval_data, 
            sample_steps=trial_config['NSTEPS'], 
            sample_offset=0) # Only doing a single sample
        
        start = datetime.now()
        forward()
        inference_time = (start - datetime.now()).total_seconds()

        start = datetime.now()
        weight_matrix*weight_matrix_compare
        reference_time = (start - datetime.now()).total_seconds()
        return inference_time/reference_time


class FPD(Objective): 
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "minimize"
    
    @staticmethod
    def failure():
        return 10e8

    @staticmethod
    def pre_process(energies, hlf_class, label): 
        """ takes hdf5_file, extracts high-level features, appends label, returns array """
        E_layer = []
        for layer_id in hlf_class.GetElayers():
            E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        EC_etas = []
        EC_phis = []
        Width_etas = []
        Width_phis = []
        for layer_id in hlf_class.layersBinnedInAlpha:
            EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
            EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
            Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
            Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
        E_layer = np.concatenate(E_layer, axis=1)
        EC_etas = np.concatenate(EC_etas, axis=1)
        EC_phis = np.concatenate(EC_phis, axis=1)
        Width_etas = np.concatenate(Width_etas, axis=1)
        Width_phis = np.concatenate(Width_phis, axis=1)
        ret = np.concatenate([np.log10(energies), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                            Width_etas/1e2, Width_phis/1e2, label*np.ones_like(energies)], axis=1)
        return ret

    @staticmethod
    def __call__(trained_model, eval_data, kwargs) -> float:

        binning_dataset = trained_model.config.get("BIN_FILE", "binning_dataset.xml")
        particle = trained_model.config.get("PART_TYPE", "photon")

        hlf = HLF.HighLevelFeatures(particle, filename=binning_dataset)
        reference_hlf = HLF.HighLevelFeatures(particle, filename=binning_dataset)
        reference_shower = []
        reference_energy = []
        for energy, _, data in eval_data: 
            reference_shower.append(data)
            reference_energy.append(energy)

        reference_shower = np.concatenate(reference_shower)
        reference_energy = np.concatenate(reference_energy)

        generated, energies = trained_model.generate(
            data_loader=eval_data, 
            sample_steps=trained_model.config.get("NSTEPS"), 
            sample_offset=0
        )

        hlf.CalculateFeatures(generated)
        reference_hlf.CalculateFeatures(reference_shower)
        
        source_array = FPD.pre_process(energies, hlf, 0.)[:, :-1]
        reference_array = FPD.pre_process(reference_energy, hlf, 1.)[:, :-1]
        
        try: 
            fpd, _ = jetnet.evaluation.fpd(
                np.nan_to_num(source_array), np.nan_to_num(reference_array)
            )
        except ValueError as err:
            print(err)
            return FPD.failure()

        return fpd