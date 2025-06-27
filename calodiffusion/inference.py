from datetime import datetime
import json
import os
import click 

import numpy as np
import h5py
import torch

from calodiffusion.utils import utils
from calodiffusion.utils import HGCal_utils as hgcal_utils
import calodiffusion.utils.plots as plots
from calodiffusion.utils.utils import LoadJson
from calodiffusion.train.evaluate import FPD, CNNCompare

from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

@click.group()
@click.option("-c", "--config")
@click.option("-d", "--data-folder", default="./data/", help="Folder containing data and MC files")
@click.option("--checkpoint-folder", default="./trained_models/", help="Folder to save checkpoints")
@click.option("-n", "--n-events", default=-1, type=int, help="Number of events to load")
@click.option("--job-idx", default=-1, type=int, help="Split generation among different jobs")
@click.option("--layer-only/--no-layer", default=False, help="Only sample layer energies")
@click.option("--reclean/--no-reclean", default=False, help="Redo preprocessing on loaded sample")
@click.option("--debug/--no-debug", default=False, help="Debugging options")
@click.option("--hgcal/--no-hgcal", default=None, is_flag=True, help="Use HGCal settings (overwrites config)")
@click.option("--seed", default=None, help='Set a manual seed (saved in config)')
@click.pass_context
def inference(ctx, debug, config, data_folder, checkpoint_folder, layer_only, job_idx, n_events, reclean, hgcal, seed): 
    ctx.ensure_object(dotdict)
    
    ctx.obj.config = LoadJson(config) if config is not None else {}
    ctx.obj.checkpoint_folder = checkpoint_folder
    ctx.obj.data_folder = data_folder
    ctx.obj.debug = debug
    ctx.obj.job_idx = job_idx
    ctx.obj.nevts = n_events 
    ctx.obj.layer_only = layer_only
    ctx.obj.reclean = reclean

    if seed is None: 
        seed = int(np.random.default_rng().integers(low=100, high=10**5))

    ctx.obj.seed = seed
    ctx.obj.config['SEED'] = seed
    if hgcal is not None: 
        ctx.obj.config['HGCAL'] = hgcal
        ctx.obj.hgcal = hgcal
    else: 
        ctx.obj.hgcal = ctx.obj.config.get("HGCAL", False)

@inference.group()
@click.option("-g", "--generated", default="", help="Path for generated shower results")
@click.option("--sample-steps", default=400, type=int, help="How many steps for sampling (override config)")
@click.option("--sample-offset", default=0, type=int, help="Skip some iterations in the sampling (noisiest iters most unstable)")
@click.option("--sample-algo", default="DDim", help="Algorithm for sampling the model output")
@click.option("--sparse-decoding", default=False, is_flag=True, help="Sampling during HGCal decoding step to reduce sparsity")
@click.option("--train-sampler/--no-train-sampler", default=None, help="For samplers requiring pre-training, train them (overwrites config)")
@click.option("--model-loc", default=None, help="Specific folder for loading existing model")
@click.option("--run-metrics", default=False, is_flag=True, help="Run metrics after sampling")
@click.pass_context
def sample(ctx, generated, sample_steps, sample_algo, sample_offset, sparse_decoding, train_sampler, model_loc, run_metrics):
    ctx.obj.config['SAMPLER'] = sample_algo
    if "SAMPLER_OPTIONS" not in ctx.obj.config.keys(): 
        ctx.obj.config['SAMPLER_OPTIONS'] = {}
    if train_sampler is not None: 
        ctx.obj.config['SAMPLER_OPTIONS']["TRAIN_SAMPLER"] =  train_sampler

    if model_loc is None: 
        raise ValueError("model-loc is required")
    
    ctx.obj.model_loc = model_loc
    ctx.obj.sample_steps = sample_steps
    ctx.obj.sample_algo = sample_algo 
    ctx.obj.sample_offset = sample_offset
    ctx.obj.sparse_decoding = sparse_decoding
    ctx.obj.generated = generated
    ctx.obj.run_metrics = run_metrics

    non_config = dotdict({key: value for key, value in ctx.obj.items() if key!='config'})
    ctx.obj.config['flags'] = non_config

@sample.command()
@click.option("--layer-model", required=True)
@click.pass_context
def layer(ctx, layer_model): 
    ctx.obj.config['layer_model'] = layer_model
    run_inference(ctx.obj, ctx.obj.config, model=lambda flags, config, load_data: TrainLayerModel(flags, config, load_data, inference=True))

@sample.command()
@click.pass_context
def diffusion(ctx):
    non_config = dotdict({key: value for key, value in ctx.obj.items() if key!='config'})
    ctx.obj.config['flags'] = non_config
    run_inference(ctx.obj, ctx.obj.config, model=TrainDiffusion)

@inference.command()
@click.option("-g", "--generated", default="", help="Path to existing generated results")
@click.option("--plot-label", default="", help="Labels for the plot")
@click.option("--plot-folder", default="./plots", help="Folder to save results")
@click.option("--plot-reshape/--no-plot-reshape", default=False, help="Plot the embedded space")
@click.option("-e", "--extension", help="Types of files to save under.", multiple=True, default=["png"])
@click.option("--cms/--no-cms", default=False, help='Use the CMS plotting style')
@click.option("--energy-min", default=-1.0, type=float, help="Min cell energy threshold")
@click.option("--geant-only", default=False, is_flag=True, help="Plots only of geant distribution")
@click.pass_context
def plot(ctx, generated, plot_label, plot_folder, plot_reshape, extension, cms, energy_min, geant_only):
    ctx.obj.plot_label = plot_label
    ctx.obj.plot_folder = plot_folder
    ctx.obj.plot_reshape = plot_reshape
    ctx.obj.plot_extensions = extension
    ctx.obj.generated = generated
    ctx.obj.cms = cms
    ctx.obj.EMin = energy_min
    ctx.obj.geant_only = geant_only

    flags = ctx.obj
    data_dict, energies = process_data_dict(flags, config=ctx.obj.config)

    plot_results(flags, ctx.obj.config, data_dict, energies)


def process_data_dict(flags, config): 
    dataset_num = config.get("DATASET_NUM", 2)

    geom_conv = None
    NN_embed = None 
    if flags.hgcal: 
        shape_embed = config.get("SHAPE_FINAL")
        NN_embed = hgcal_utils.HGCalConverter(bins=shape_embed, geom_file=config["BIN_FILE"])
        if flags.plot_reshape: 
            NN_embed.init()
    
    elif dataset_num <= 1: 
        bins = utils.XMLHandler(config["PART_TYPE"], config["BIN_FILE"])
        NN_embed = utils.GeomConverter(bins)


    if(not flags.geant_only):
        generated, energy = LoadSamples(flags.generated, flags, config, geom_conv, NN_embed=NN_embed)
        total_events = generated.shape[0]

    data = []
    energies = []

    eval_files = utils.get_files(config["EVAL"], folder=flags.data_folder)
    for dataset in eval_files:
        print(dataset)
        showers, energy = LoadSamples(dataset, flags, config, geom_conv, NN_embed)
        data.append(showers)
        energies.append(energy)

        total_events = 0
        for d in data:
            total_events += d.shape[0]
        if total_events >= flags.nevts: 
            break

    if len(data) == 0: 
        raise ValueError("No Evaluation Data passed, please change the `EVAL` field of the config")
    
    energies = np.concatenate(energies)
    data_dict = {
        "Geant4": np.concatenate(data),
    }

    if not flags.geant_only: 
        data_dict[utils.name_translate(generated_file_path=flags.generated)] = generated

    return data_dict, energies


def write_out(fout, flags, config, generated, energies, first_write = True, do_mask = False):

    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    dataset_num = config.get("DATASET_NUM", 2)

    if not orig_shape: 
        generated = generated.reshape(config["SHAPE_ORIG"])

    energies = np.reshape(energies,(energies.shape[0],-1))

    hgcal = config.get("HGCAL", False)
    shower_scale = config.get('SHOWERSCALE', 200.)

    if (do_mask) and (dataset_num > 1):
        #mask for voxels that are always empty
        mask_file = os.path.join(flags.data_folder,config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
        if(not os.path.exists(mask_file)):
            print("Creating mask based on data batch")
            mask = np.sum(generated, 0) == 0

        else:
            with h5py.File(mask_file,"r") as h5f:
                mask = h5f['mask'][:]
        generated = generated *(np.reshape(mask,(1,-1))==0)

    generated = np.reshape(generated, config['SHAPE_ORIG'])
    if first_write :
        print("Creating {}".format(fout))
        shape = list(config['SHAPE_ORIG'])
        shape[0] = None
        if not hgcal:
            with h5py.File(fout,"w") as h5f:
                h5f.create_dataset("showers", data= (1./shower_scale) * generated,  compression = 'gzip', maxshape = shape, chunks = True)
                h5f.create_dataset("incident_energies", data=(1./shower_scale) *energies, compression = 'gzip', maxshape = (None, 1), chunks = True)
        else:

            with h5py.File(fout,"w") as h5f:
                h5f.create_dataset("showers", data=(1./shower_scale)* generated, compression = 'gzip',maxshape=shape, chunks = True)
                h5f.create_dataset("gen_info", data=energies, compression = 'gzip', maxshape = (None, energies.shape[1]), chunks = True)
    else:
        print("Appending to {}".format(fout))
        with h5py.File(fout,"a") as h5f:
            if not hgcal:
                utils.append_h5(h5f, 'showers', (1./shower_scale) * generated)
                utils.append_h5(h5f, 'incident_energies', (1./shower_scale) * energies)
            else:
                utils.append_h5(h5f, 'showers', (1./shower_scale) * generated)
                utils.append_h5(h5f, 'gen_info', energies)


def LoadSamples(fp, flags, config, geom_conv, NN_embed=None):
    print("Loading " + fp)
    end = None if flags.nevts < 0 else flags.nevts
    shower_scale = config.get("SHOWERSCALE", 0.001)

    if(config.get("DATASET_NUM", 2) <= 1): 
        flags.plot_reshape = True

    if (not flags.hgcal) or flags.plot_reshape:
        shape_plot = config["SHAPE_FINAL"]
    else:
        shape_plot = config["SHAPE_PAD"]

    with h5py.File(fp, "r") as h5f:
        if flags.hgcal:
            generated = h5f["showers"][:end, :, : config["MAX_CELLS"]] * shower_scale
            energies = h5f["gen_info"][:end, 0]
        else:
            generated = h5f["showers"][:end] * shower_scale
            energies = h5f["incident_energies"][:end] * shower_scale

    energies = np.reshape(energies, (-1, 1))
    if flags.plot_reshape:
        if config.get("DATASET_NUM", 2) <= 1:
            generated = NN_embed.convert(NN_embed.reshape(generated)).detach().numpy()
        elif flags.hgcal:
            generated = torch.from_numpy(generated.astype(np.float32)).reshape(
                config["SHAPE_PAD"]
            )
            generated = NN_embed.enc(generated).detach().numpy()

    if(flags.plot_reshape or (not flags.hgcal)):
        generated = np.reshape(generated, shape_plot)

    if flags.EMin > 0.0:
        mask = generated < flags.EMin
        generated = utils.apply_mask_conserveE(generated, mask)

    return generated, energies


def plot_results(flags, config, data_dict, energies): 
    plot_routines = {
        "Energy per layer": plots.ELayer(flags, config),
        "Energy": plots.HistEtot(flags, config),
        "2D Energy scatter split": plots.ScatterESplit(flags, config),
        "Energy Ratio split": plots.HistERatio(flags, config),
        "Layer Sparsity": plots.SparsityLayer(flags, config),
    }

    if flags.hgcal and not flags.plot_reshape:
        plot_routines.update(
            {
                "Energy R": plots.RadialEnergyHGCal(flags, config),
                "Energy R Center": plots.RCenterHGCal(flags, config),
                "Energy Phi Center": plots.PhiCenterHGCal(flags, config),
                "Nhits": plots.HistNhits(flags, config),
                "Max voxel": plots.HistMaxELayer(flags, config),
                "VoxelE": plots.HistVoxelE(flags, config),
            }
        )

    elif not flags.layer_only:
        plot_routines.update(
            {
                "Nhits": plots.HistNhits(flags, config),
                "VoxelE": plots.HistVoxelE(flags, config),
                "Shower width": plots.AverageShowerWidth(flags, config),
                "Max voxel": plots.HistMaxELayer(flags, config),
                "Energy per radius": plots.AverageER(flags, config),
                "Energy per phi": plots.AverageEPhi(flags, config),
            }
        )

    if (not config["CYLINDRICAL"]) and (
        config["SHAPE_PAD"][-1] == config["SHAPE_PAD"][-2]
    ):
        plot_routines["2D average shower"] = plots.Plot_Shower_2D(flags, config)

    for plotting_method in plot_routines.values():
        plotting_method(data_dict, energies)

def run_metrics(flags, model, generated, eval_data): 
    results = {}

    # Run FDP (if installed)
    try: 
        metric = FPD(
            binning_dataset=flags.config.get("BIN_FILE", "binning_dataset.xml"), 
            particle=flags.config.get("PART_TYPE", "photon"), 
            hgcal=flags.config.get("HGCAL", False)
        )
        fpd = metric(trained_model=model, eval_data=eval_data)
        results["FPD"] = float(fpd)
    except ImportError: 
        print("WARNING: Jetnet not installed, cannot run FPD")

    # Run CNN Compare (if model is here/correctly trained for the network)
    try: 
        metric = CNNCompare(trained_model=model, config=flags.config, flags=flags)
        cnn_sep = metric(eval_data=eval_data)
        results["CNN"] = float(cnn_sep)
    except Exception as e:
        print(f"WARNING: Unable to run CNN evaluation metric: {e}")
        raise Exception(e)

    try: 
        ""
    except: 
        ""

    return results


def run_inference(flags, config, model):
    data_loader, _ = utils.load_data(flags, config, eval=True)

    model_instance = model(flags, config, load_data=False)
    model_instance.init_model()
    model, _, _, _, _, _  = model_instance.pickup_checkpoint(
        model=model_instance.model,
        optimizer=None,
        scheduler=None,
        early_stopper=None,
        n_epochs=0,
        restart_training=True,
    )
    sample_steps = flags.sample_steps if flags.sample_steps is not None else config.get("SAMPLE_STEPS", 400)

    generated, energies = model.generate(data_loader, sample_steps, flags.debug, flags.sample_offset, sparse_decoding=flags.sparse_decoding)
    if(flags.generated == ""):
        fout = f"{model_instance.checkpoint_folder}/generated_{config['CHECKPOINT_NAME']}_{flags.sample_algo}{sample_steps}_{datetime.now().timestamp()}.h5"
    else: 
        fout = flags.generated
    write_out(fout, flags, config, generated, energies, first_write=True)

    if flags.run_metrics:
        results = run_metrics(flags, model_instance.model, generated, data_loader)

        with open(f"{model_instance.checkpoint_folder}/metrics.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    inference()
