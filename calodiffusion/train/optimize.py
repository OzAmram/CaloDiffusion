from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable, Literal, Sequence, Union

import numpy as np
import torch
import optuna
import json
import os

from datetime import datetime
from calodiffusion.train import Train
from calodiffusion.utils import utils
from calodiffusion.train import evaluate


class Optimize: 
    """
    
    Optimize the training results and sampling quality. 
    Assume a config the form of: 
    {
        <Generic Unchanged Settings, datasets, etc>, 
        "OPTIMIZE": {
            "setting_1": [float_min, float_max], 
            "setting_2": [int_min, int_max], 
            "setting_3": [str_choice_1, str_choice_2, str_choice_3 ...]
            "setting_4: [True, False], 
            "SAMPLER_SETTINGS": {
                <Settings for each type of sampler. Few each sampler docs for details>
            }
        }
    }

    Exceptions - 
    * LAYER_SIZE_UNET (u-net architecture settings) - becomes a dictionary with the integer list fields: 
        {
            "init_unet": The range of the initial layer size, 
            "n_unet_layers": Range for up/down operations in the unet, 
            "layer_ratio": Range for how small the inner layer is compared to init_unet
        }
    * SAMPLER_SETTINGS (where SAMPLER="Restart") - has the following custom fields to create "RESTART_LIST": 
        {
            "RESTART_I": Integer range for number of possible restart configurations, 
            "N_RESTART": Number of steps each restart configuration can take, 
            "RESTART_K": Integer range of K parameter, 
            "RESTART_T": Range allowed for T_MIN_{i}. T_MAX_{i} is decided by taking t_min_i as the bottom of the range and t_min_i + {top of the restart_t range} as the top
        }
    """
    def __init__(self, flags, trainer: type[Train], objectives: Literal["COUNT", "FPD", "CNN"]) -> None:

        implemented_objectives: dict[str, type[Objective]] = {
            "COUNT": Count(), 
            "FPD": FPD(), 
            "CNN": CNNMetric()
        }
        self.flags = flags
        self.trainer = trainer

        self.objectives = []
        if isinstance(objectives, str):
            objectives = [objectives]
        for objective in objectives: 
            self.objectives.append(implemented_objectives[objective])

    def train(self, trial_config): 
        config = self.suggest_config(trial_config)
        train_model = self.trainer(flags=self.flags, config=config, save_model=False)
        model, _, _ = train_model.train()
        eval_data = train_model.loader_val
        return model, eval_data, config

    def suggest_config(self, trial_config): 
        if isinstance(self.flags.config, str): 
            config = utils.LoadJson(self.flags.config)
        else: 
            config = self.flags.config 

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
            elif key not in ("SAMPLER", "SAMPLER_SETTINGS"): 
                if all([isinstance(i, str) for i in values]) or (True in values): 
                    config[key] = trial_config.suggest_categorical(key, values)
                elif all([isinstance(i, int) for i in values]): 
                    config[key] = trial_config.suggest_int(key, *values)       
                else: 
                    config[key] = trial_config.suggest_float(key, *values)
            else: 
                config = self.suggest_sampler_config(config, trial_config)

        return config

    def suggest_hyperparam(self, setting_name, config, hyparam_settings, trial_config, type_=float): 
        if setting_name in hyparam_settings.keys(): 
            if type_ is float: 
                config[setting_name] = trial_config.suggest_float(setting_name, *hyparam_settings[setting_name])
            elif type_ is int: 
                config[setting_name] = trial_config.suggest_int(setting_name, *hyparam_settings[setting_name])
            else: 
                config[setting_name] = trial_config.suggest_categorical(setting_name, hyparam_settings[setting_name])
        return config 
    
    def suggest_sampler_config(self, config, trial_config): 
        optimized_section = config.get("OPTIMIZE", {})
        sampler = config.get('SAMPLER')
        if not sampler: 
            sampler = trial_config.suggest_categorical("SAMPLER", optimized_section.get("SAMPLER", []))
            config["SAMPLER"] = sampler

        sampler_config = defaultdict(dict)
        sampler_settings = optimized_section.get("SAMPLER_SETTINGS", {})

        # For each kind of sampler there are different hyperparams. Each need to be handled differently
        # Samplers without hyperparams: ["DDIM", "DDPM", "DPMPP2M"]: 

        if sampler in ["DPM", "DPMPPSDE", "DPMPP2S", "DPMPP2MSDE", "DPMAdaptive", "DPMPP3MSDE", "Restart"]:
            sampler_config = self.suggest_hyperparam("ETA", sampler_config, sampler_settings, trial_config)
            sampler_config = self.suggest_hyperparam("S_NOISE", sampler_config, sampler_settings, trial_config)

        if sampler == "DPMAdaptive": 
            config = self.suggest_hyperparam("ORDER", sampler_config, sampler_settings, trial_config, type_=int)
            for setting in ["R_TOL", "A_TOL", "H_INIT", "T_ERROR", "ACCEPT_SAFETY"]: 
                sampler_config = self.suggest_hyperparam(setting, sampler_config, sampler_settings, trial_config)

        if sampler == 'DPMPPSDE': 
            sampler_config = self.suggest_hyperparam("R", sampler_config, sampler_settings, trial_config)

        if sampler == 'DPMPP2MSDE': 
            sampler_config = self.suggest_hyperparam("SOLVER", sampler_config, sampler_settings, trial_config, type_=str)

        if sampler in ["LMS", "Euler", "Heun", "DPM2", "Restart"]: 
            # Samplers in EDM class
            sampler_config = self.suggest_hyperparam("NOISY_SAMPLE", sampler_config, sampler_settings, trial_config, type_=str)
            sampler_config = self.suggest_hyperparam("ORIG_SCHEDULE", sampler_config, sampler_settings, trial_config, type_=str)
            if sampler_config.get("ORIG_SCHEDULE", True): 
                sampler_config = self.suggest_hyperparam("C1", sampler_config, sampler_settings, trial_config)

            sampler_config = self.suggest_hyperparam("RHO", sampler_config, sampler_settings, trial_config, type_=int)
            sampler_config = self.suggest_hyperparam("SIGMA_MIN", sampler_config, sampler_settings, trial_config)

            if sampler in ["Euler", "Heun", "DPM2", "Restart"]: 
                sampler_config = self.suggest_hyperparam("S_MIN", sampler_config, sampler_settings, trial_config)
                sampler_config = self.suggest_hyperparam("S_MAX", sampler_config, sampler_settings, trial_config)
                sampler_config = self.suggest_hyperparam("S_NOISE", sampler_config, sampler_settings, trial_config)
                sampler_config = self.suggest_hyperparam("S_CHURN", sampler_config, sampler_settings, trial_config)
        
        if sampler == "LMS": 
            sampler_config = self.suggest_hyperparam("ORDER", sampler_config, sampler_settings, trial_config, type_=int)

        if sampler == "Restart": 
            sampler_config = self.suggest_hyperparam("RESTART_GAMMA", sampler_config, sampler_settings, trial_config)
            sampler_config = self.suggest_hyperparam("C2", sampler_config, sampler_settings, trial_config)

            sampler_config = self.suggest_hyperparam("RESTART_I", sampler_config, sampler_settings, trial_config, type_=int)
            sampler_config = self.suggest_hyperparam("N_RESTART", sampler_config, sampler_settings, trial_config, type_=int)
            n_restart = sampler_config.get("N_RESTART", 4)
            restart_settings = {}
            for num in range(sampler_config.get('RESTART_I', 4)): 
                k_i = trial_config.suggest_int(f"RESTART_K_{num}", *sampler_settings.get("RESTART_K", [1, 10]))
                restart_t_range = sampler_settings.get("RESTART_T", [0.01, 50])
                t_min_i = trial_config.suggest_float(f"RESTART_T_MIN_{num}", *restart_t_range)
                t_max_i = trial_config.suggest_float(f"RESTART_T_MAX_{num}", t_min_i, t_min_i+restart_t_range[-1])
                restart_settings[str(num)] = [n_restart, k_i, t_min_i, t_max_i]
            sampler_config['RESTART_LIST'] = restart_settings
        config['SAMPLER_SETTINGS'] = sampler_config
        return config

    def eval(self, model, eval_data, config) -> Sequence: 
        config['flags'] = self.flags
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

    def save_results(self, study): 
        study_items = dict(study.trials_dataframe())
        study_results = {} 
        for key, value in study_items.items(): 
            study_results[key] = value.to_list()

        save_loc = self.flags.results_folder
        if not os.path.exists(save_loc): 
            os.makedirs(save_loc)

        report_path = f"{save_loc.rstrip('/')}/{self.flags.study_name}_report.json"
        with open(report_path, 'a') as f: 
            json.dump(study_results, f, default=str)


    def __call__(self) -> None:
        study = optuna.create_study(
            study_name=self.flags.study_name, 
            load_if_exists=True, 
            directions=[obj.direction() for obj in self.objectives]
            )
        study.optimize(
            self.objective, 
            n_trials=self.flags.n_trials, 
            timeout=300
        )
        self.save_results(study)
        

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
    def __call__(trained_model, eval_data, kwargs) -> float:

        binning_dataset = trained_model.config.get("BIN_FILE", "binning_dataset.xml")
        particle = trained_model.config.get("PART_TYPE", "photon")

        fpd_calc = evaluate.FDP(binning_dataset, particle)
        
        try: 
            return fpd_calc(trained_model, eval_data, kwargs)
        except evaluate.FDPCalculationError:
            return FPD.failure()
        
class CNNMetric(Objective):
    @staticmethod
    def failure(): 
        return 1 
    
    @staticmethod
    def direction() -> Literal['minimize', 'maximize']:
        return "maximize"
    
    @staticmethod
    def __call__(trained_model, eval_data, kwargs):
        cnn_method = evaluate.CNNCompare(
            trained_model=trained_model, 
            config= kwargs, 
            flags = kwargs['flags']
        )
        return cnn_method(eval_data)
