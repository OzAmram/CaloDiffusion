"""
Tests for all of the optimizer options. 
Unless explicitly stated, these tests only investigate learning rate or sampler type.
Failure indicates the command line interface is broken, not the optimization itself. 
The converse is also true - a successful test does not indicate the optimization is working.
"""

import pytest 
import os
import shutil

from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.utils import utils


class MockMethod(): 
    def __init__(self):
        pass

    def state_dict(self): 
        return []

@pytest.fixture(scope="module")
def diffusion_weights(config): 
    checkpoint_folder = "./test_optimization/"
    name = "diffusion_weights"

    def diffusion_factory(hgcal:bool = False): 
        mock_config = config({
            "CHECKPOINT_NAME": "mock_weights" , 
        })
        args = utils.dotdict({
            "checkpoint": checkpoint_folder, 
            "data_folder": "", 
            "hgcal": hgcal
        })

        t = TrainDiffusion(args, utils.LoadJson(mock_config), load_data=False, save_model=True)
        t.save(
            model_state=t.model.model.state_dict, 
            epoch=0, 
            name=name, 
            training_losses=[], 
            validation_losses=[], 
            scheduler=MockMethod(),
            optimizer=MockMethod(),
            early_stopper=MockMethod()
        )
        return os.path.join(t.checkpoint_folder, f"{name}.pth")
    
    yield diffusion_factory

    shutil.rmtree(checkpoint_folder)

@pytest.fixture(scope="module")
def layer_weights(config): 
    checkpoint_folder = "./test_optimization/"
    name = "layer_weights"

    def diffusion_factory(hgcal:bool = False): 
        mock_config = config({
            "CHECKPOINT_NAME": "mock_weights" , 
        })
        args = utils.dotdict({
            "checkpoint": checkpoint_folder, 
            "data_folder": "", 
            "hgcal": hgcal
        })

        t = TrainLayerModel(args, utils.LoadJson(mock_config), load_data=False, save_model=True)
        t.save(
            model_state=t.model.model.state_dict, 
            epoch=0, 
            name=name, 
            training_losses=[], 
            validation_losses=[], 
            scheduler=MockMethod(),
            optimizer=MockMethod(),
            early_stopper=MockMethod()
        )
        return os.path.join(t.checkpoint_folder, f"{name}.pth")
    
    yield diffusion_factory

    shutil.rmtree(checkpoint_folder)

def test_optimize_training_diffusion(execute, config, pytestconfig): 
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{}
    })
    command = f"python3 calodiffusion/optimize.py \
        --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training diffusion"
    
    exit = execute(command)
    assert exit == 0

def test_optimize_sampler_diffusion(execute, config, weights, pytestconfig): 
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{}
    })
    command = f"python3 calodiffusion/optimize.py \
        --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            sample --model-loc {weights} diffusion"
    exit = execute(command)
    assert exit == 0

def test_optimize_training_diffusion_hgcal(execute, config, pytestconfig):
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{"LR": [0.001, 0.0001]}
    })
    command = f"python3 calodiffusion/optimize.py \
        --hgcal --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training diffusion"
    exit = execute(command)
    assert exit == 0

def test_optimize_sampler_diffusion_hgcal(execute, config, pytestconfig):
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}
    })
    command = f"python3 calodiffusion/optimize.py \
        --hgcal --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training layer"
    exit = execute(command)
    assert exit == 0

def test_optimize_train_layer(): 
    pass

def test_optimize_sampler_layer(): 
    pass

def test_optimize_train_layer_hgcal(): 
    pass

def test_optimize_sampler_layer_hgcal(): 
    pass



def test_count_obj(): 
    ""

def test_cnn_obj(): 
    ""

def test_fpd_obj(): 
    ""

def test_separation_obj(): 
    ""