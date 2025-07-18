
import json
import pytest
import os
import argparse

from calodiffusion.models.calodiffusion import CaloDiffusion
from calodiffusion.models.layerdiffusion import LayerDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.train.train_diffusion import TrainDiffusion

from calodiffusion.utils import utils
import torch

@pytest.mark.hgcal
@pytest.mark.parametrize("model_type", [CaloDiffusion, LayerDiffusion])
def test_pushforward_sampling_layer(config, hgcal_data, pytestconfig, model_type): 
    """Test the base pushforward inference sampling with hgcal """
    data_file = hgcal_data("imm_test.h5")
    
    config_file = config({
        "FILES": [data_file],
        "EVAL": [data_file],
        "CHECKPOINT_NAME": "imm_test",
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""",
        'SHAPE_ORIG': [-1, 28, 1988],
        'DATASET_NUM': 111,
        'SHAPE_PAD': [-1, 1, 28, 12, 21],
        'SHAPE_FINAL': [-1, 1, 28, 12, 21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET': [32, 32, 64, 96],
        'SHOWER_EMBED': 'NN-pre-embed',
        'HGCAL': True,
        'SAMPLER': 'Pushforward',
        'SAMPLER_OPTIONS': {
            'P_MEAN': 0.1,
            'P_STD': 0.5,
        }
    })
    
    # Load config
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    flags = argparse.Namespace(**{
        "data_folder": pytestconfig.getoption("data_dir"),
        "checkpoint_folder": "./testing_checkpoints/",
        'nevts': 10,
        "hgcal": True, 
        "seed": 42
    })

    # Create model and sampler
    model = model_type(config_dict, n_steps=2)    
    # Test pushforward sampling
    data,_ = utils.load_data(flags, config_dict)
    for E, layers, x in data:
        generated, xs, x0s =  model.sampler_algorithm(
            model=model,
            start=x,
            energy=E,
            layers=layers,
            num_steps=5,
            sample_offset=0,
            debug=False
        )
        
    # Check outputs
    assert generated is not None
    assert generated.shape == x.shape

@pytest.mark.hgcal
@pytest.mark.parametrize("model_type", [CaloDiffusion, LayerDiffusion])
def test_imm_loss(hgcal_data, pytestconfig, config, model_type): 
    """Verify you can calculate the IMM Loss for different model types"""
    data_file = hgcal_data("imm_training.h5")
    
    config_file = config({
        "FILES": [data_file],
        "EVAL": [data_file],
        "CHECKPOINT_NAME": "imm_training",
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""",
        'SHAPE_ORIG': [-1, 28, 1988],
        'DATASET_NUM': 111,
        'SHAPE_PAD': [-1, 1, 28, 12, 21],
        'SHAPE_FINAL': [-1, 1, 28, 12, 21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET': [32, 32, 64, 96],
        'SHOWER_EMBED': 'NN-pre-embed',
        'HGCAL': True,
        'TRAINING_OBJ': 'IMM',
        'SAMPLER': 'Pushforward',
        'BATCH': 8,
        'NSTEPS': 10
    })
    c = json.load(open(config_file, 'r'))

    flags = argparse.Namespace(**{
        "data_folder": pytestconfig.getoption("data_dir"),
        "checkpoint_folder": "./testing_checkpoints/",
        'nevts': 10,
        "hgcal": True, 
        "seed": 42, 
        "model": model_type.__name__,
    })

    model = model_type(c, n_steps=5)
    model.init_model()

    train_loader, _ = utils.load_data(flags, c, eval=False)
    for energy, layers, data in train_loader:
        loss = model.loss_function(model, data=data, E=energy, layers=layers)
        
        # Check loss
        assert loss is not None
        assert loss.requires_grad
        assert not torch.isnan(loss).any()
        assert loss.detach().numpy().all() >= 0


@pytest.mark.hgcal
@pytest.mark.parametrize("model_trainer", [TrainLayerModel, TrainDiffusion])
def test_imm_training(hgcal_data, pytestconfig, config, model_trainer): 
    """Test IMM training with pushforward sampling"""
    data_file = hgcal_data("imm_training.h5")
    
    config_file = config({
        "FILES": [data_file],
        "EVAL": [data_file],
        "CHECKPOINT_NAME": "imm_training",
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""",
        'SHAPE_ORIG': [-1, 28, 1988],
        'DATASET_NUM': 111,
        'SHAPE_PAD': [-1, 1, 28, 12, 21],
        'SHAPE_FINAL': [-1, 1, 28, 12, 21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET': [32, 32, 64, 96],
        'SHOWER_EMBED': 'NN-pre-embed',
        'HGCAL': True,
        'TRAINING_OBJ': 'IMM',
        'SAMPLER': 'Pushforward',
        'BATCH': 8,
        'NSTEPS': 10
    })
    
    # Load config
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
            
    flags = argparse.Namespace(**{
        "data_folder": pytestconfig.getoption("data_dir"),
        "checkpoint_folder": "./testing_checkpoints/",
        'nevts': 10,
        "hgcal": True,
        "seed": 42, 
        "model": model_trainer.__name__,
        "config": config_file,
        "load": False

    })
    
    trainer = model_trainer(flags, config_dict)
    trainer.train()

@pytest.mark.hgcal
def test_imm_pipeline(config, hgcal_data, pytestconfig, execute): 
    """Test IMM sampler through CLI training and inference"""
    data_file = hgcal_data("imm_inference.h5")
    data_dir = pytestconfig.getoption("data_dir")
    
    # Train a model first
    train_config = config({
        "FILES": [data_file],
        "EVAL": [data_file],
        "CHECKPOINT_NAME": "imm_inference",
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""",
        'SHAPE_ORIG': [-1, 28, 1988],
        'DATASET_NUM': 111,
        'SHAPE_PAD': [-1, 1, 28, 12, 21],
        'SHAPE_FINAL': [-1, 1, 28, 12, 21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET': [32, 32, 64, 96],
        'SHOWER_EMBED': 'NN-pre-embed',
        'HGCAL': True,
        'SAMPLER': 'Pushforward',
        'NSTEPS': 5,
        'MAXEPOCH': 1, 
        "TRAINING_OBJ": "IMM",
    })
    
    # Train model
    train_command = f"calodif-train -c {train_config} -d {data_dir} -n 5 --hgcal --checkpoint ./testing_checkpoints/ diffusion"
    train_exit = execute(train_command)
    assert train_exit == 0
    assert os.path.exists("./testing_checkpoints/imm_inference_Diffusion/final.pth")


    inference_command = f"calodif-inference -c {train_config} -d {data_dir} -n 5 --checkpoint-folder ./testing_checkpoints/ sample --sample-steps 2 --model-loc ./testing_checkpoints/imm_inference_Diffusion/final.pth diffusion"
    inference_exit = execute(inference_command)
    assert inference_exit == 0