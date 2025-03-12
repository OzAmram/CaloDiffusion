"""
Tests for all of the optimizer options. 
Unless explicitly stated, these tests only investigate learning rate or sampler type.
Failure indicates the command line interface is broken, not the optimization itself. 
The converse is also true - a successful test does not indicate the optimization is working.
"""

import pytest 

@pytest.fixture(scope="module")
def outdir(): 
    return ""

@pytest.fixture(scope="module")
def weights(): 
    return ""

def test_no_param_training_opt(execute, config, pytestconfig): 
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{}
    })
    command = f"python3 calodiffusion/optimize.py \
        --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training diffusion"
    exit = execute(command)
    assert exit == 0

def test_no_param_sampler_opt(execute, config, pytestconfig): 
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{}
    })
    command = f"python3 calodiffusion/optimize.py \
        --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            sample diffusion"
    exit = execute(command)
    assert exit == 0

def test_hgcal_opt(execute, config, pytestconfig):
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{"LR": [0.001, 0.0001]}
    })
    command = f"python3 calodiffusion/optimize.py \
        --hgcal --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training diffusion"
    exit = execute(command)
    assert exit == 0

def test_hgcal_layer_opt(execute, config, pytestconfig):
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{"SAMPLER": ["DDIM", "DDPM"]}
    })
    command = f"python3 calodiffusion/optimize.py \
        --hgcal --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training layer"
    exit = execute(command)
    assert exit == 0

def test_hgcal_sampler_opt(execute, config, pytestconfig):
    config = config({
        "CHECKPOINT_NAME": "no_param_test" , 
        "OPTIMIZE":{"SAMPLER": [0.001, 0.0001]}
    })
    command = f"python3 calodiffusion/optimize.py \
        --hgcal --n-trials 2 -n 30 --config {config} --data-dir {pytestconfig.getoption("data_dir")} \
            training layer"
    exit = execute(command)
    assert exit == 0

def tes_hgcal_sampler_layer_opt(execute, config, pytestconfig):
    ""

def test_sampler_opt(execute): 
    ""

def test_sampler_layer_opt(execute): 
    ""

def test_training_opt(execute): 
    ""

def test_training_layer_opt(execute): 
    ""

def test_count_obj(): 
    ""

def test_cnn_obj(): 
    ""

def test_fpd_obj(): 
    ""

def test_separation_obj(): 
    ""