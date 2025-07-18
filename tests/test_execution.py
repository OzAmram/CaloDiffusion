import os
import pytest 


@pytest.fixture(scope="session")
def pion_config(config): 
    def pion_config(checkpoint_name): 
        return config({
        'FILES':['dataset_1_pions_1.hdf5'],
        'EVAL':['dataset_1_pions_1.hdf5'],
        'BIN_FILE': "./CaloChallenge/code/binning_dataset_1_pions.xml",
        'PART_TYPE' : 'pion',
        'AVG_SHOWER_LOC' : "./CaloChallenge/dataset_2_avg_showers.hdf5",
        'DATASET_NUM' : 0,
        'SHAPE_ORIG':[-1,533],
        'SHAPE_PAD':[-1,1, 533],
        'SHAPE_FINAL':[-1,1,7,10,23],
        'BATCH':128,
        'NLAYERS':3,
        'LAYER_SIZE_AE':[32,64, 64,32],
        'DIM_RED_AE':[0,2, 0, 2],
        'LAYER_SIZE_UNET' : [16, 16, 16, 32],
        'COND_SIZE_UNET' : 128, 
        "CHECKPOINT_NAME": checkpoint_name})
    
    yield pion_config

@pytest.mark.dependency() 
def test_train_diffusion(config, execute, pytestconfig): 
    config = config()
    data_dir = pytestconfig.getoption("data_dir")

    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.pion
@pytest.mark.dependency() 
def test_train_diffusion_pion(pion_config, execute, pytestconfig): 
    config = pion_config("pion_test")
    data_dir = pytestconfig.getoption("data_dir")

    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.dependency() 
def test_train_layer(config, execute, pytestconfig): 
    config = config({"CHECKPOINT_NAME": "layer", "SHAPE_PAD": [-1,1,5,10,30]})
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ layer"
    exit = execute(command)
    assert exit == 0


@pytest.mark.pion
def test_train_layer_pion(pion_config, execute, pytestconfig): 
    config = pion_config("pion_test_layer")
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-train -c {config} -d {data_dir} -n 10 --checkpoint ./testing_checkpoints/ layer"
    exit = execute(command)
    assert exit == 0

@pytest.mark.hgcal
@pytest.mark.dependency() 
def test_train_hgcal(config, execute, pytestconfig, hgcal_data): 
    data_dir = pytestconfig.getoption("data_dir")
    data_file = hgcal_data("mock_hgcal.h5")
    config = config({
        "FILES":[data_file],
        "EVAL":[data_file],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""", 
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
def test_inference_diffusion(config, execute, pytestconfig): 
    config = config()
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0


@pytest.mark.pion
@pytest.mark.dependency(depends=["test_train_diffusion_pion"]) 
def test_inference_diffusion_pion(pion_config, execute, pytestconfig): 
    config = pion_config("pion_test")
    data_dir = pytestconfig.getoption("data_dir")
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/pion_test_Diffusion/final.pth\
                diffusion"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_train_diffusion", "test_train_layer"]) 
def test_inference_layer(config,execute, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    config = config()
    command = f"calodif-inference -c {config} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
            sample --sample-steps 2 --model-loc ./testing_checkpoints/photon_test_Diffusion/final.pth\
                layer --layer-model ./testing_checkpoints/layer_LayerModel/final.pth"
    exit = execute(command)
    assert exit == 0

@pytest.mark.hgcal
@pytest.mark.dependency(depends=["test_train_hgcal"]) 
def test_inference_hgcal(config, execute, pytestconfig, hgcal_data): 
    data_file = hgcal_data("mock_hgcal.h5")
    config = config({
        "FILES":[data_file],
        "EVAL":[data_file],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""", 
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
def test_plotting_diffusion(config, execute, pytestconfig): 
    data_dir = pytestconfig.getoption("data_dir")
    base_dir = "./testing_checkpoints/photon_test_Diffusion"
    generated = [f for f in os.listdir(base_dir) if "generated" in f][0]
    command = f"calodif-inference -c {config()} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
        plot -g {base_dir}/{generated}  --plot-folder ./testing_checkpoints/plots/"
    exit = execute(command)
    assert exit == 0

@pytest.mark.dependency(depends=["test_inference_diffusion"]) 
def test_plotting_geant(config, execute, pytestconfig):
    data_dir = pytestconfig.getoption("data_dir")
    base_dir = "./testing_checkpoints/photon_test_Diffusion"
    generated = [f for f in os.listdir(base_dir) if "generated" in f][0]
    command = f"calodif-inference -c {config()} -d {data_dir} -n 10 --checkpoint-folder ./testing_checkpoints/\
        plot --plot-folder ./testing_checkpoints/plots/ --geant-only -g {base_dir}/{generated} "
    exit = execute(command)
    assert exit == 0

@pytest.mark.hgcal
@pytest.mark.dependency(depends=["test_inference_hgcal"]) 
def test_plotting_hgcal(config, execute, pytestconfig, hgcal_data): 
    data_dir = pytestconfig.getoption("data_dir")
    data_file = hgcal_data("mock_hgcal.h5")
    config = config({
        "FILES":[data_file],
        "EVAL":[data_file],
        "CHECKPOINT_NAME": "hgcal", 
        "BIN_FILE": f"""{pytestconfig.getoption("hgcalshowers")}/HGCalShowers/geoms/geom_william.pkl""", 
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
