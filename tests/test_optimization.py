"""
Tests for all of the optimizer options. 
Unless explicitly stated, these tests only investigate learning rate or sampler type.
Failure indicates the command line interface is broken, not the optimization itself. 
The converse is also true - a successful test does not indicate the optimization is working.
"""

import pytest
import shutil
import os
from calodiffusion.utils import utils

@pytest.fixture(scope='function')
def global_settings(study_name, pytestconfig): 
    folder = "./mock_opt_study/"
    def settings(config): 
        return f"--n-trials 1 -n 10 -o COUNT --results-folder {folder} --name {study_name} --config {config} --data-folder {pytestconfig.getoption("data_dir")}", folder

    yield settings 

    try: 
        shutil.rmtree(folder)
    except FileNotFoundError: 
        pass

@pytest.fixture(scope="function")
def study_name(): 
    return "mock_study"

def test_optimize_training_diffusion(execute, config, global_settings, study_name): 
    config = config({
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"LR": [0.0001, 0.001]}
    })
    settings, folder = global_settings(config)
    command = f"python3 calodiffusion/optimize.py \
        {settings} \
        train diffusion"
    exit = execute(command)
    assert exit == 0
    
    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)

def test_optimize_sampler_diffusion(execute, config, diffusion_weights, global_settings, study_name): 
    config = config({
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"LR": [0.0001, 0.001]}
    })
    settings, folder = global_settings(config)
    command = f"python3 calodiffusion/optimize.py \
        {settings} \
        sample --model-loc {diffusion_weights()} diffusion"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)


def test_optimize_training_diffusion_hgcal(execute, config, global_settings, study_name, hgcal_data, pytestconfig):
    data_file = hgcal_data("mock_hgcal.h5")
    
    config = config({
        "FILES":[data_file], 
        "FILES_EVAL": [data_file],
        "EVAL": [data_file], 
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}, 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
        'HGCAL': True
    
    })

    settings, folder = global_settings(config)
    command = f"python3 calodiffusion/optimize.py \
        {settings} --hgcal \
            train diffusion"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)

def test_optimize_sampler_diffusion_hgcal(execute, config, diffusion_weights, global_settings, study_name, pytestconfig, hgcal_data): 
    data_file = hgcal_data("mock_hgcal.h5")
    
    config = config({
        "FILES":[data_file], 
        "FILES_EVAL": [data_file],
        "EVAL": [data_file], 
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}, 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
        'HGCAL': True

    })

    settings, folder = global_settings(config)

    command = f"python3 calodiffusion/optimize.py \
        {settings} --hgcal \
        sample --model-loc {diffusion_weights(hgcal=True)} diffusion"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)


def test_optimize_train_layer(config, execute, global_settings): 
    config = config({
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"LR": [0.0001, 0.001]}
    })
    command = f"python3 calodiffusion/optimize.py \
        {global_settings(config)}\
            train layer"
    
    exit = execute(command)
    assert exit == 0

def test_optimize_sampler_layer(config, diffusion_weights, layer_weights, global_settings, execute): 
    config = config({
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}
    })
    command = f"python3 calodiffusion/optimize.py \
        {global_settings(config)}\
        sample --model-loc {diffusion_weights()} layer --layer-model {layer_weights()}"
    
    exit = execute(command)
    assert exit == 0

def test_optimize_sampler_layer_hgcal(hgcal_data, config, pytestconfig, study_name, diffusion_weights, global_settings, layer_weights, execute): 
    data_file = hgcal_data("mock_hgcal.h5")
    
    config = config({
        "FILES":[data_file], 
        "FILES_EVAL": [data_file],
        "EVAL": [data_file], 
        "CHECKPOINT_NAME": "opt_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}, 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
        'HGCAL': True

    })

    settings, folder = global_settings(config)

    command = f"python3 calodiffusion/optimize.py \
        {settings} --hgcal \
        sample --model-loc {diffusion_weights(hgcal=True)} layer --layer-model {layer_weights(hgcal=True)}"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)


def test_correct_training_settings(config, study_name, global_settings, execute): 
    settings_dict = {
        "HGCAL": False,
        "OPTIMIZE":{
            "LR": [0.0001, 0.001], 
            "TRAINING_OBJ" : ["hybrid_weight", "noise_pred", "mean_pred"],
            "LOSS_TYPE" : ["huber", "l1", "l2"],
            "TIME_EMBED" : ["sigma", "sin", "id", "log"],
            "SHOWER_EMBED": ["orig-NN", "NN-pre-embed"],
            "NLAYERS":[2,5],
            }
        }
    settings, folder = global_settings(config(settings_dict))
    command = f"python3 calodiffusion/optimize.py {settings} --no-hgcal train diffusion"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)
    report = utils.LoadJson(report)
    params = {key.removeprefix("params_") for key in report.keys() if "params_" in key}
    assert params == set(settings_dict['OPTIMIZE'].keys())

def test_correct_sampler_settings(config, study_name, global_settings, execute, diffusion_weights): 
    settings_dict = {
        "OPTIMIZE":{
        "SAMPLER": ["DPM", "DDim", "DDPM", "Restart", "DPM2", "Heun", "Euler", "LMS", "DPMPP3MSDE", "DPMPP2MSDE", "DPMPP2M", "DPMPPSDE", "DPMPP2S"],
        "SAMPLER_SETTINGS": {
            "ETA": [0.0, 1.1], 
            "S_NOISE": [0.0, 1.1], 
            "ORDER":[1, 6],
            "R": [0.0, 2.5], 
            "SOLVER": ["huen", "midpoint"], 
            "NOISY_SAMPLE": [True, False], 
            "ORIG_SCHEDULE": [True, False], 
            "C1": [0.0, 0.1], 
            "C2": [0.0, 0.1], 
            "RHO": [2, 10], 
            "SIGMA_MIN": [0.0, 0.5], 
            "S_MIN": [0.0, 0.5], 
            "S_NOISE": [0.5, 2.5], 
            "RESTART_GAMMA": [0.01, 0.5], 
            "RESTART_I": [1, 5], 
            "N_RESTART": [1, 5], 
            "RESTART_K": [1, 10], 
            "RESTART_T": [0.01, 50]
        }        }
    }
    settings, folder = global_settings(config(settings_dict))
    command = f"python3 calodiffusion/optimize.py {settings} --no-hgcal sample --model-loc {diffusion_weights()} diffusion"
    exit = execute(command)
    assert exit == 0

    report = os.path.join(folder, study_name, "report.json")
    assert os.path.exists(report)
    report = utils.LoadJson(report)
    params = {key.removeprefix("params_") for key in report.keys() if "params_" in key}
    assert params != {}
    assert params.issubset(set(settings_dict['OPTIMIZE'].keys()).union(set(settings_dict["OPTIMIZE"]["SAMPLER_SETTINGS"].keys())))