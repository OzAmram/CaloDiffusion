import numpy as np
from calodiffusion.train.optimize import EvalCNNMetric, EvalCount, EvalMeanSeparation, EvalFPD, EvalLoss
from calodiffusion.train.evaluate import CNNCompare, HistogramSeparation
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.utils import utils
import pytest 
import time

##### Objectives from optimization #####

def test_count_obj(config, pytestconfig, checkpoint_folder): 
    """
    Approximate how fast a model is compared to a basic matrix operation
    """
    random = np.random.default_rng()
    metric = EvalCount()
    args = utils.dotdict({"data_folder": pytestconfig.getoption("data_dir"), "checkpoint_folder": checkpoint_folder, 'nevts': 10})
    c = config({"CHECKPOINT_NAME": "test_count_obj"})
    model = TrainDiffusion(args, utils.LoadJson(c), load_data=True, save_model=False)
    

    model.init_model()
    operation_time = metric(model.model, eval_data=model.loader_train, config=utils.LoadJson(c))
    start = time.process_time()
    random.random((24, 24))*random.random((24, 24))
    compare_time = abs(start - time.process_time())
    assert operation_time > compare_time


def test_cnn_obj(checkpoint_folder, config, pytestconfig, hgcal_data): 
    # Only implemented for dataset 2 and 3 - needs to be adapted for hgcal
    # Cannot be tested until that is put in place
    metric = EvalCNNMetric()
    # Test Photon 

    args = utils.dotdict({
        "data_folder": pytestconfig.getoption("data_dir"), 
        "results_folder": checkpoint_folder, 
        "checkpoint_folder": checkpoint_folder, 
        'nevts': 10, 
        "hgcal": True})
    
    c = config({
                "FILES": [hgcal_data("hgcal_test.hdf5")],
                "CHECKPOINT_NAME": "test_cnn_obj_hgcal",
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
    
    model = TrainDiffusion(args, utils.LoadJson(c), load_data=True, save_model=False)
    model.init_model()
    c = utils.LoadJson(c)
    c['flags'] = args

    cnn_result = metric(model.model, eval_data=model.loader_train, config=c)

    assert cnn_result is not None # TODO what's the range?

@pytest.mark.xfail    
def test_fpd_obj(checkpoint_folder, config, pytestconfig): 
    metric = EvalFPD()
    args = utils.dotdict({"data_folder": pytestconfig.getoption("data_dir"), "checkpoint_folder": checkpoint_folder, 'nevts': 10})
    
    c = config({"CHECKPOINT_NAME": "test_fpd_obj_photon"})
    model = TrainDiffusion(args, utils.LoadJson(c), load_data=True, save_model=False)
    fpd = metric(model.model, eval_data=model.loader_train, config=config)
    assert fpd

    # TODO Fix the import error - figure out installation 
    

def test_separation_obj(checkpoint_folder, config, pytestconfig): 
    metric = EvalMeanSeparation()
    args = utils.dotdict({"data_folder": pytestconfig.getoption("data_dir"), "checkpoint_folder": checkpoint_folder, 'nevts': 10})
    
    c = config({"CHECKPOINT_NAME": "test_separation_obj"})
    model = TrainDiffusion(args, utils.LoadJson(c), load_data=True, save_model=False)
    model.init_model()
    # Default behavior doesn't need to be configured - just works
    # Default is the energy
    energy_sep = metric(model.model, eval_data=model.loader_train, config=utils.LoadJson(c))
    assert energy_sep

    # Can use a different separation histogram - here use sparsity
    c = utils.LoadJson(
        config({
            "CHECKPOINT_NAME": "test_separation_obj", 
            "SEPARATION_METRIC": "HistEtot"
            })
    )
    sparity_sep = metric(model.model, eval_data=model.loader_train, config=c)
    
    assert sparity_sep
    assert sparity_sep != energy_sep

    # Can handle multiple metrics
    c = utils.LoadJson(
        config({
            "CHECKPOINT_NAME": "test_separation_obj", 
            "SEPARATION_METRIC": ["HistNhits", "HistVoxelE"]
            })
    )
    mean_sep = metric(model.model, eval_data=model.loader_train, config=c)
    
    assert mean_sep


def test_loss_obj(checkpoint_folder, config, pytestconfig): 
    metric = EvalLoss()
    args = utils.dotdict({"data_folder": pytestconfig.getoption("data_dir"), "checkpoint_folder": checkpoint_folder, 'nevts': 10})
    
    c = config({"CHECKPOINT_NAME": "test_loss_obj"})
    model = TrainDiffusion(args, utils.LoadJson(c), load_data=True, save_model=False)
    model.init_model()
    model.train()

    loss = metric(model.model, eval_data=model.loader_train, config=utils.LoadJson(c))
    assert loss is not None
    assert loss > 0  # Loss value varies on random noise uses - cannot compare to the training loss

    # Can load in a different loss metric
    model.config = utils.LoadJson(
        config({
            "CHECKPOINT_NAME": "test_separation_obj", 
            "LOSS_METRIC": "noise_pred"
        })
    )
    new_loss = metric(model.model, eval_data=model.loader_train, config=c)
    assert new_loss != loss 
    assert new_loss > 0
#########################################

##### Basic Metric Calculations #########


def test_histogram_metric(pytestconfig, checkpoint_folder, config, hgcal_data): 
    args = utils.dotdict({
        "data_folder": pytestconfig.getoption("data_dir"), 
        "checkpoint_folder": checkpoint_folder, 
        "nevts": 10, 
        "results_folder": checkpoint_folder})
    
    config = config({
        "CHECKPOINT_NAME": "test_histogram_metric",                 
        "FILES":["dataset_1_photons_1.hdf5"]})
    config = utils.LoadJson(config)

    for metric_name in ["HistERatio","HistEtot","HistNhits", "HistVoxelE", "HistMaxE"]: 
        metric = HistogramSeparation(metric=metric_name)
        data, _ = utils.load_data(args, config, eval=False)
        energy, _, original = next(iter(data))
        sep = metric(original, generated=original, energies=energy)

        assert isinstance(sep, float), f"Problem with metric {metric_name}"

    # then doing the hgcal exclusive metrics
    for metric_name in ["RCenterHGCal","PhiCenterHGCal"]: 
        metric = HistogramSeparation(
            metric=metric_name, 
            bin_file=f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl")
        config['FILES'] = [hgcal_data("histogram_test.hdf5")]
        config.update({
            'SHAPE_ORIG': [-1,28,1988],
            'DATASET_NUM' : 111,
            'SHAPE_PAD':[-1,1,28,12,21],
            'SHAPE_FINAL':[-1,1,28,12,21],
            'MAX_CELLS': 1988,
            'LAYER_SIZE_UNET' : [32, 32, 64, 96], 
            "HGCAL": True
        })
        args.hgcal=True
        data, _ = utils.load_data(args, config, eval=False)
        energy, _, original = next(iter(data))
        sep = metric(original, generated=original, energies=energy)

        assert isinstance(sep, float), f"Problem with metric {metric_name}"


def test_average_histogram_metrics(pytestconfig, checkpoint_folder, config, hgcal_data): 
    args = utils.dotdict({
        "data_folder": pytestconfig.getoption("data_dir"), 
        "checkpoint_folder": checkpoint_folder, 
        "nevts": 10, 
        "results_folder": checkpoint_folder})
    
    config = config({
        "CHECKPOINT_NAME": "test_histogram_metric",                 
        "FILES":["dataset_1_photons_1.hdf5"]})
    config = utils.LoadJson(config)

    data, _ = utils.load_data(args, config, eval=False)
    energy, _, original = next(iter(data))

    metric = HistogramSeparation(metric=["HistERatio","HistEtot","HistNhits"])
    sep = metric(original, generated=original, energies=energy)
    assert isinstance(sep, float)

    metric = HistogramSeparation(metric=["HistERatio","HistEtot","HistNhits", "HistVoxelE"])
    sep = metric(original, generated=original, energies=energy)
    assert isinstance(sep, float)


