import pytest 
import os
import subprocess
import json 
import shutil

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
@pytest.fixture(scope='function')
def execute(): 
    def run_execution(command:str): 
        command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
        process = subprocess.run(command, capture_output=True)
        print(process.stdout.decode())
        print(process.stderr.decode())
        return process.returncode
    yield run_execution
    
@pytest.fixture(scope="session")
def config(pytestconfig): 
    # setup
    checkpoint_dir = "./testing_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True) # Okay to overwrite if there was a failure that caused teardown not to be triggered
    
    def config_factory(extra_settings:dict={}): 
        fp = f"{checkpoint_dir}test_config.json"
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

    # Teardown
    shutil.rmtree(checkpoint_dir)
