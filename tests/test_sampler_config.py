import pytest

from calodiffusion.train.optimize import Optimize
from calodiffusion.train.train_diffusion import Diffusion


class MockSuggester: 
    def suggest_int(*args, **kwargs): 
        return 1

    def suggest_float(*args, **kwargs): 
        return 1.
    
    def suggest_categorical(name, *args, **kwargs): 
        return args[0]

class Flags: 
    def __init__(self, config):
        self.nevts = 10
        self.config = config
        self.data_folder = "./data/"
        self.frac = 0.85
        self.load = False
        self.sample = True

setting_fields = [
    ("DDim", []), 
    ("DDPM", []), 
    ("DPM", ["ETA", "S_NOISE"]), 
    ("DPMPP3MSDE", ["ETA", "S_NOISE"]), 
    ("DPMPPSDE", ["R"]), 
    ("DPMPP2MSDE", ["ETA", "S_NOISE", {"name":"SOLVER", "option":["heun", "midpoint"]}]), 
    ("LMS", [ {"name":"NOISY_SAMPLE", "option":[True, False]}, {"name":"ORIG_SCHEDULE", "option":[True, False]}, "C1", "RHO", "SIGMA_MIN", "ORDER"]), 
    ("Euler", [ {"name":"NOISY_SAMPLE", "option":[True, False]}, {"name":"ORIG_SCHEDULE", "option":[True, False]}, "C1", "RHO", "SIGMA_MIN", "S_MIN", "S_MAX", "S_NOISE", "S_CHURN"]),
    ("Heun", [{"name":"NOISY_SAMPLE", "option":[True, False]}, {"name":"ORIG_SCHEDULE", "option":[True, False]}, "C1", "RHO", "SIGMA_MIN", "S_MIN", "S_MAX", "S_NOISE", "S_CHURN"]),
    ("DPM2", [ {"name":"NOISY_SAMPLE", "option":[True, False]}, {"name":"ORIG_SCHEDULE", "option":[True, False]}, "C1", "RHO", "SIGMA_MIN", "S_MIN", "S_MAX", "S_NOISE", "S_CHURN"]),
    ("Restart", [ {"name":"NOISY_SAMPLE", "option":[True, False]}, {"name":"ORIG_SCHEDULE", "option":[True, False]}, "C1", "RHO", "SIGMA_MIN", "RESTART_GAMMA", "C2", "RESTART_I", "N_RESTART"]),
    ("DPMAdaptive", ["ORDER", "R_TOL", "A_TOL", "H_INIT", "T_ERROR", "ACCEPT_SAFETY"])
]

@pytest.mark.parametrize("sampler_name,options", setting_fields)
def test_sampler_setups(sampler_name, options): 
    sampler_options = {}
    for option in options: 
        if not isinstance(option, dict): 
            sampler_options[option] = [0, 10]
        else: 
            sampler_options[option['name']] =  option["option"]

    flags = Flags({
        "SAMPLER": sampler_name,
        "NSTEPS": 10, 
        "LOSS_TYPE": "mse",
        "BATCH": 256,
        "PART_TYPE" : "pion",
        "DATASET_NUM" : 0,
        "TIME_EMBED" : "sigma",
        "SHOWERMAP": "layer-logit-norm",
        "FILES":["dataset_1_pions_1.hdf5"],
        "EVAL":["dataset_1_pions_1.hdf5"],
        "BIN_FILE": "../CaloDiffusion/CaloChallenge/code/binning_dataset_1_pions.xml",
        "SHAPE_ORIG":[-1,533],
        "SHAPE":[-1,7,10,23,1],
        "SHAPE_PAD":[-1,1,7,10,23],
        "EMAX":4194.304,
        "EMIN":0.256,
        "ECUT":0.0000001,
        "MAXEPOCH":1,
        "EARLYSTOP": 2,
        "LR": 4e-4,
        "NLAYERS":3,
        "MAXDEP":3.1,
        "logE": True,
        "COND_EMBED" : "id",
        "SHOWER_EMBED" : "orig-NN",
        "COND_SIZE_UNET": 128, 
        "LAYER_SIZE_UNET":[16, 16, 16, 32], 

        "OPTIMIZE": {
            "SAMPLER_SETTINGS": sampler_options
        }
    })
    opt = Optimize(flags, trainer=None, objectives=[])

    trial = MockSuggester()

    config = opt.suggest_config(trial)
    config_sampler = config['SAMPLER_SETTINGS']
    for option in options: 
        if isinstance(option, str):
            assert option in config_sampler.keys(), f"Missing {option} for {sampler_name}"
            assert config_sampler[option] == 1, f"Option {option} == {config_sampler[option]}"

    trainer = Diffusion(flags, flags.config, save_model=False)
    model, _, _ = trainer.train()
    assert model.generate(
            trainer.loader_val,             
            sample_steps=trainer.config.get("NSTEPS"), 
            sample_offset=0
        ) is not None

