import pytest
import torch
import json
import os


from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.utils import utils

def generate_test_parameters():
    model_types = ["layer", "diffusion"]
    data_types = ["calochallenge", "hgcal"]
    sampler_names = [
        "DDim", "DDPM", "DPM", "DPMAdaptive", "DPMPP2S", "DPMPPSDE",
        "DPMPP2M", "DPMPP2MSDE", "DPMPP3MSDE", "LMS", "Euler",
        "Heun", "DPM2", "Restart", "Consistency"
    ]
    test_params = []
    for model_type in model_types:
        for data_type in data_types:
            for sampler_name in sampler_names:
                test_params.append((model_type, data_type, sampler_name))
    return test_params

@pytest.mark.parametrize("model_type, data_type, sampler_name", generate_test_parameters())
def test_sampler(
    config, 
    model_type, 
    data_type, 
    sampler_name, 
    hgcal_data,
    checkpoint_folder, 
    pytestconfig): 


    if model_type == "layer": 
        model = TrainLayerModel
    elif model_type == "diffusion":
        model = TrainDiffusion

    c = {"SAMPLER": sampler_name}
    if data_type == "hgcal": 
        c.update({
            "FILES": [hgcal_data(name="mock_hgcal.h5")],
            "EVAL": [hgcal_data(name="mock_hgcal.h5")],
            "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl", 
            'SHAPE_ORIG': [-1,28,1988],
            'DATASET_NUM' : 111,
            'SHAPE_PAD':[-1,1,28,12,21],
            'SHAPE_FINAL':[-1,1,28,12,21],
            'MAX_CELLS': 1988,
            'LAYER_SIZE_UNET' : [32, 32, 64, 96],
            'SHOWER_EMBED' : 'NN-pre-embed',
            "HGCAL": True
        })
        
    c = config(c)

    flags = utils.dotdict(
        dict(
            config=c, 
            data_folder=pytestconfig.getoption("data_dir"), 
            checkpoint_folder=checkpoint_folder, 
            nevts=10, 
            hgcal=True if data_type == "hgcal" else False
        )
    )

    model_instance = model(flags, utils.LoadJson(c), load_data=True)
    model_instance.init_model()

    generated, energies = model_instance.model.generate(model_instance.loader_train, 2, False, 0)
    assert generated is not None
    assert energies is not None
    assert len(generated) == 10 # 10 samples generated
