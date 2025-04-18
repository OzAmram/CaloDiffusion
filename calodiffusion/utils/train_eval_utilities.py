from calodiffusion.train.evaluate import CNNCompare
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel

from calodiffusion.utils import utils

def train_cnn_comparison(config, data_folder, model_loc, hgcal=True, layer_model_loc=None, layer_model=False): 
    config = utils.LoadJson(config)
    if hgcal: 
        config["HGCAL"] = True

    args = utils.dotdict({
        "load": True, 
        "data_folder": data_folder,
        "results_folder": "./",
        "hgcal": hgcal,
        "checkpoint_folder": "./", 
        "model_loc": model_loc, 
        "layer_model": layer_model_loc

    })

    config["RETRAIN_EVAL_NETWORK"] = True

    if layer_model:
        model = TrainLayerModel(args, config, load_data=False, save_model=False, inference=True)
    else: 
        model = TrainDiffusion(args, config, load_data=False, save_model=False)

    # When "retrain" is enabled, the model is automatically trained
    CNNCompare(trained_model=model.model, config=config, flags=args)
