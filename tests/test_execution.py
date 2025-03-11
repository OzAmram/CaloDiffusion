import os
import pytest 
import subprocess
import json 
import shutil


def execute(command:str): 
    command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
    process = subprocess.run(command, capture_output=True)
    print(process.stdout.decode())
    print(process.stderr.decode())
    return process.returncode

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


@pytest.mark.dependency() 
def test_train_diffusion(config, pytestconfig): 
    config = config()
    data_dir = pytestconfig.getoption("data_dir")

    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency() 
def test_train_diffusion_pion(config, pytestconfig): 
    config = config({
        'FILES':['dataset_1_pions_1.hdf5'],
        'EVAL':['dataset_1_pions_2.hdf5'],
        'BIN_FILE': "./CaloChallenge/code/binning_dataset_1_pions.xml",
        'PART_TYPE' : 'pion',
        'AVG_SHOWER_LOC' : "./CaloChallenge/dataset_2_avg_showers.hdf5",
        'DATASET_NUM' : 0,
        'SHAPE_ORIG':[-1,533],
        'SHAPE_PAD':[-1,1, 533],
        'SHAPE_FINAL':[-1,1,7,10,23],
        "CHECKPOINT_NAME": "pion_test"
    })
    data_dir = pytestconfig.getoption("data_dir")

    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency() 
def test_train_layer(config, pytestconfig): 
    config = config({"CHECKPOINT_NAME": "layer", "SHAPE_PAD": [-1,1,5,10,30]})
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ layer"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency() 
def test_train_hgcal(config, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    config = config({
        "FILES":['HGCal_showers1.h5'],
        "EVAL":['HGCal_showers1.h5'],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
    })
    command = f"calodif-train -c {config} -d {data_dir} -n 10 --hgcal --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency(depends=["test_train_diffusion"]) 
def test_inference_diffusion(config, pytestconfig): 
    config = config()
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency(depends=["test_train_diffusion_pion"]) 
def test_inference_diffusion_pion(config, pytestconfig): 
    config = config({
        'FILES':['dataset_1_pions_1.hdf5'],
        'EVAL':['dataset_1_pions_1.hdf5'],
        'BIN_FILE': f"{pytestconfig.getoption("calochallenge")}/CaloChallenge/code/binning_dataset_1_pions.xml",
        'PART_TYPE' : 'pion',
        'DATASET_NUM' : 0,
        'SHAPE_ORIG':[-1,533],
        'SHAPE_PAD':[-1,1, 533],
        'SHAPE_FINAL':[-1,1,7,10,23]
    })
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/pion_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency(depends=["test_train_diffusion", "test_train_layer"]) 
def test_inference_layer(config, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    config = config()
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                layer --layer-model ./testing_checkpoints/layer_LayerModel/final.pth"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency(depends=["test_train_hgcal"]) 
def test_inference_hgcal(config, pytestconfig): 
    config = config({
        "FILES":['HGCal_showers1.h5'],
        "EVAL":['HGCal_showers1.h5'],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
    })
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/ --hgcal\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/hgcal_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency(depends=["test_inference_diffusion"]) 
def test_plotting_diffusion(config, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    base_dir = "./testing_checkpoints/photon_test_Diffusion"
    generated = [f for f in os.listdir(base_dir) if "generated" in f][0]
    command = f"calodif-inference -c {config()} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
        plot -g {base_dir}/{generated}  --plot-folder ./testing_checkpoints/plots/"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_inference_diffusion"]) 
def test_plotting_geant(config, pytestconfig):
    data_dir = pytestconfig.getoption("data_dir")
    base_dir = "./testing_checkpoints/photon_test_Diffusion"
    generated = [f for f in os.listdir(base_dir) if "generated" in f][0]
    command = f"calodif-inference -c {config()} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
        plot --plot-folder ./testing_checkpoints/plots/ --geant-only -g {base_dir}/{generated} "
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_inference_hgcal"]) 
def test_plotting_hgcal(config, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    config = config({
        "FILES":['HGCal_showers1.h5'],
        "EVAL":['HGCal_showers1.h5'],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geom_william.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
    })
    base_dir = "./testing_checkpoints/hgcal_Diffusion"
    generated = [f for f in os.listdir(base_dir) if "generated" in f][0]
    command = f"calodif-inference --hgcal -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
        plot --plot-folder ./testing_checkpoints/plots/ -g {base_dir}/{generated}"
    exit = execute(command)
    assert exit == 0