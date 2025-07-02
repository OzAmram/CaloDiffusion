from datetime import datetime
import pytest 
import os
import subprocess
import json 
import shutil

from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.utils import utils

def pytest_addoption(parser):
    parser.addoption("--data-dir", default='./data/', help='Add a specific data dir to use during tests')
    parser.addoption("--calochallenge", default='.', help='Add a specific dir for the CaloChallenge dir is located')
    parser.addoption("--hgcalshowers", default='.', help='Add a specific dir for the HGCalShowers dir is located')

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hgcal: Test only runs on hgcal data (deselect with -m 'not hgcal')"
    )
    config.addinivalue_line(
        "markers", "pion: Test only runs on pion dataset (deselect with -m 'not pion')"
    )


class MockMethod(): 
    def __init__(self):
        pass

    def state_dict(self): 
        return []


@pytest.fixture(scope='session')
def checkpoint_folder(): 

    checkpoint_dir = "./tests/checkpoints/"
    yield checkpoint_dir

    shutil.rmtree(checkpoint_dir)


@pytest.fixture(scope='function')
def execute(): 
    def run_execution(command:str): 
        command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
        process = subprocess.run(command, capture_output=True)
        print(process.stdout.decode())
        print(process.stderr.decode())
        return process.returncode
    yield run_execution


@pytest.fixture(scope="function")
def hgcal_data(pytestconfig): 
    import h5py
    import numpy as np
    
    # make fake hgcal data
    data_dir = pytestconfig.getoption("data_dir")
    hgcal_dir_name = "hgcal_data/"
    hgcal_dir = os.path.join(data_dir, hgcal_dir_name)
    os.makedirs(hgcal_dir, exist_ok=True)

    def hgcal_factory(name:str): 
        f_name = os.path.join(hgcal_dir, name)
        if os.path.exists(f_name): 
            os.remove(f_name)
        with h5py.File(f_name, "w") as f:
            f.create_dataset("gen_info", shape=(360, 3), dtype="<f4")
            f['gen_info'][:] = np.random.rand(360, 3).astype("<f4")

            f.create_dataset("showers", shape=(360, 28, 3000), dtype="<f4")
            f['showers'][:] = np.random.rand(360, 28, 3000).astype("<f4")

            f.create_dataset("incident_energies", shape=(840000, 1, 1), dtype="<f4")
            f['incident_energies'][:] = np.random.rand(840000, 1, 1).astype("<f4")

        return os.path.join(hgcal_dir_name, name)
    
    yield hgcal_factory

    shutil.rmtree(hgcal_dir)

@pytest.fixture(scope="session")
def config(pytestconfig, checkpoint_folder): 
    # setup
    os.makedirs(checkpoint_folder, exist_ok=True) # Okay to overwrite if there was a failure that caused teardown not to be triggered
    
    def config_factory(extra_settings:dict={}): 
        fp = f"{checkpoint_folder}test_config_{datetime.now().timestamp()}.json"
        config = {
            "FILES":["dataset_1_photons_1.hdf5"],
            "EVAL":["dataset_1_photons_1.hdf5"],
            "BIN_FILE": f"{pytestconfig.getoption("calochallenge")}/CaloChallenge/code/binning_dataset_1_photons.xml",
            "EMBED":128,
            "EMAX":4194.304,
            "EMIN":0.256,
            "ECUT":0.0000001,
            "logE":True,
            "PART_TYPE" : "photon",
            "DATASET_NUM" : 1,
            "HOLDOUT" : 0,
            "SHAPE_ORIG":[-1,368],
            "SHAPE":[-1,5,10,30,1],
            "SHAPE_PAD":[-1,1,5,10,30],
            "SHAPE_FINAL":[-1,1,5,10,30],
            "BATCH":128,
            "LR":4e-4,
            "MAXEPOCH":2,
            "NLAYERS":3,
            "EARLYSTOP":20,
            "MAXDEP":3.1,
            "LAYER_SIZE_AE":[32,64, 64,32],
            "DIM_RED_AE":[0, 2, 0, 2],
            "LAYER_SIZE_UNET" : [32, 32, 64, 96],
            "COND_SIZE_UNET" : 128,
            "KERNEL":[3,3,3],
            "STRIDE":[3,2,2],
            "BLOCK_ATTN" : True,
            "MID_ATTN" : True,
            "COMPRESS_Z" : True,
            "CYLINDRICAL": True,
            "SHOWERMAP": "layer-logit-norm",
            "PHI_INPUT": True,
            "R_Z_INPUT": True,
            "BETA_MAX" : 0.02,
            "NOISE_SCHED": "log",
            "NSTEPS": 100,
            "CONSIS_NSTEPS": 100,
            "COLD_DIFFU" : False,
            "COLD_NOISE" : 1.0,
            "TRAINING_OBJ" : "hybrid_weight",
            "LOSS_TYPE" : "huber",
            "TIME_EMBED" : "sigma",
            "COND_EMBED" : "id",
            "SHOWER_EMBED" : "orig-NN",
            "SAMPLER": "DDim", 
            "CHECKPOINT_NAME": "photon_test", 
            "COLD_DIFFU" : False
        }
        config.update(extra_settings)

        with open(fp, "w") as f: 
            json.dump(config, f)

        return fp 
    
    yield config_factory


@pytest.fixture(scope="module")
def diffusion_weights(config, checkpoint_folder, pytestconfig): 
    name = "diffusion_weights"

    def diffusion_factory(hgcal:bool = False): 
        config_settings = {
            "CHECKPOINT_NAME": "mock_weights" , 
        }
        if hgcal:
            config_settings.update({
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
        mock_config = config(config_settings)
        args = utils.dotdict({
            "checkpoint_folder": checkpoint_folder, 
            "data_folder": "", 
            "hgcal": hgcal
        })
        t = TrainDiffusion(args, utils.LoadJson(mock_config), load_data=False, save_model=True)
        t.init_model()
        t.save(
            model_state=t.model.state_dict(), 
            epoch=0, 
            name=name, 
            training_losses={}, 
            validation_losses={}, 
            scheduler=MockMethod(),
            optimizer=MockMethod(),
            early_stopper=MockMethod()
        )
        return os.path.join(t.checkpoint_folder, f"{name}.pth")
    
    yield diffusion_factory


@pytest.fixture(scope="module")
def layer_weights(config, checkpoint_folder, pytestconfig): 
    name = "layer_weights"

    def layer_factory(hgcal:bool = False): 
        config_settings = {
            "CHECKPOINT_NAME": "mock_weights" , 
        }
        if hgcal:
            config_settings.update({
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

        mock_config = config(config_settings)

        args = utils.dotdict({
            "checkpoint_folder": checkpoint_folder, 
            "data_folder": "", 
            "hgcal": hgcal
        })

        t = TrainLayerModel(args, utils.LoadJson(mock_config), load_data=False, save_model=True)
        t.init_model()
        t.save(
            model_state=t.model.state_dict(), 
            epoch=0, 
            name=name, 
            training_losses={}, 
            validation_losses={}, 
            scheduler=MockMethod(),
            optimizer=MockMethod(),
            early_stopper=MockMethod()
        )
        return os.path.join(t.checkpoint_folder, f"{name}.pth")
    
    yield layer_factory