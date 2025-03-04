import pytest 
import subprocess
import json 


def execute(command:str): 
    command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
    process = subprocess.run(command, capture_output=True)
    print(process.stdout.decode())
    print(process.stderr.decode())
    return process.returncode

def make_config(extra_settings:dict={}): 
    fp = "./test_config.json"
    # Uses a photon by default
    config = {
        "FILES":["dataset_1_photons_1.hdf5"],
        "EVAL":["dataset_1_photons_1.hdf5"],
        "BIN_FILE": "./CaloChallenge/code/binning_dataset_1_photons.xml",
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

@pytest.mark.dependency() 
def test_train_diffusion(): 
    config = make_config()
    command = f"calodif-train -c {config} -d ./data/ -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency() 
def test_train_diffusion_pion(): 
    config = make_config({
        'FILES':['dataset_1_pions_1.hdf5'],
        'EVAL':['dataset_1_pions_2.hdf5'],
        'BIN_FILE': "./CaloChallenge/code/binning_dataset_1_pions.xml",
        'PART_TYPE' : 'pion',
        'AVG_SHOWER_LOC' : "./CaloChallenge/dataset_2_avg_showers.hdf5",
        'DATASET_NUM' : 0,
        'SHAPE_ORIG':[-1,533],
        'SHAPE':[-1,7,10,23,1],
        'SHAPE_PAD':[-1,1,7,10,23],
    })
    command = f"calodif-train -c {config} -d ./data/ -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency() 
def test_train_layer(): 
    config = make_config({"CHECKPOINT_NAME": "layer", "SHAPE_PAD": [-1,1,5,10,30]}) # TODO logic to use either shape-pad or shape-final based on what is present
    command = f"calodif-train -c {config} -d ./data/ -n 10 --checkpoint ./testing_checkpoints/ layer"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency() 
def test_train_hgcal(): 
    config = make_config({
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": "./HGCalShowers/geom.pkl", 
        'SHAPE_ORIG': [-1,28,1988],
        'DATASET_NUM' : 111,
        'SHAPE_PAD':[-1,1,28,12,21],
        'SHAPE_FINAL':[-1,1,28,12,21],
        'MAX_CELLS': 1988,
        'LAYER_SIZE_UNET' : [32, 32, 64, 96],
        'SHOWER_EMBED' : 'NN-pre-embed',
    })
    command = f"calodif-train -c {config} -d ./data/ -n 10 --hgcal --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_train_diffusion"]) 
def test_inference_diffusion(): 
    config = make_config()
    command = f"calodif-inference -c {config} -d ./data/ -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_train_diffusion_pion"]) 
def test_inference_diffusion_pion(): 
    config = make_config()
    command = f"calodif-inference -c {config} -d ./data/ -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_train_diffusion", "test_train_layer"]) 
def test_inference_layer(): 
    config = make_config()
    command = f"calodif-inference -c {config} -d ./data/ -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                layer --layer-model ./testing_checkpoints/photon_test_Diffusion/final.pth"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_train_hgcal"]) 
def test_inference_hgcal(): 
    ""

@pytest.mark.dependency(depends=["test_inference_diffusion"]) 
def test_plotting(): 
    ""