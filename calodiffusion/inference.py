from datetime import datetime
import os
import click 

import numpy as np
import h5py

from calodiffusion.utils import utils
import calodiffusion.utils.plots as plots
from calodiffusion.utils.utils import LoadJson

from calodiffusion.train import Diffusion, TrainLayerModel

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
@click.option("--seed", default=None, help='Set a manual seed (saved in config)')
@click.pass_context
<<<<<<< HEAD
def inference(ctx, debug, config, data_folder, checkpoint_folder, layer_only, job_idx, n_events): 
=======
def inference_parser(ctx, debug, config, data_folder, checkpoint_folder, layer_only, job_idx, n_events, seed): 
>>>>>>> ac78ce1 (Small corrections post merge pr)
    ctx.ensure_object(dotdict)
    
    ctx.obj.config = LoadJson(config) if config is not None else {}
    ctx.obj.checkpoint_folder = checkpoint_folder
    ctx.obj.data_folder = data_folder
    ctx.obj.debug = debug
    ctx.obj.job_idx = job_idx
    ctx.obj.nevts = n_events 
    ctx.obj.layer_only = layer_only

    if seed is None: 
        seed = int(np.random.default_rng().integers(low=100, high=10**5))

    ctx.obj.seed = seed
    ctx.obj.config['SEED'] = seed

@inference.group()
@click.option("--sample-steps", default=400, type=int, help="How many steps for sampling (override config)")
@click.option("--sample-offset", default=0, type=int, help="Skip some iterations in the sampling (noisiest iters most unstable)")
@click.option("--sample-algo", default="DDim", help="Algorithm for sampling the model output")
@click.option("--train-sampler/--no-train-sampler", default=None, help="For samplers requiring pre-training, train them (overwrites config)")
@click.option("--model-loc", default=None, help="Specific folder for loading existing model")
@click.pass_context
def sample(ctx, sample_steps, sample_algo, sample_offset, train_sampler, model_loc):
    ctx.obj.config['SAMPLER'] = sample_algo
    if "SAMPLER_OPTIONS" not in ctx.obj.config.keys(): 
        ctx.obj.config['SAMPLER_OPTIONS'] = {}
    if train_sampler is not None: 
        ctx.obj.config['SAMPLER_OPTIONS']["TRAIN_SAMPLER"] =  train_sampler
    ctx.obj.model_loc = model_loc
    ctx.obj.sample_steps = sample_steps
    ctx.obj.sample_algo = sample_algo 
    ctx.obj.sample_offset = sample_offset

    non_config = dotdict({key: value for key, value in ctx.obj.items() if key!='config'})
    ctx.obj.config['flags'] = non_config

@sample.command()
@click.option("--layer-model", required=True)
@click.pass_context
def layer(ctx, layer_model): 
    ctx.obj.config['layer_model'] = layer_model
    run_inference(ctx.obj, ctx.obj.config, model=TrainLayerModel)

@sample.command()
@click.pass_context
def diffusion(ctx):
    non_config = dotdict({key: value for key, value in ctx.obj.items() if key!='config'})
    ctx.obj.config['flags'] = non_config
    run_inference(ctx.obj, ctx.obj.config, model=Diffusion)

@inference.command()
@click.option("-g", "--generated", help="Generated showers")
@click.option("--plot-label", default="", help="Labels for the plot")
@click.option("--plot-folder", default="./plots", help="Folder to save results")
@click.option("-e", "--extension", help="Types of files to save under.", multiple=True, default=["png"])
@click.pass_context
def plot(ctx, generated, plot_label, plot_folder, extension):
    ctx.obj.plot_label = plot_label
    ctx.obj.plot_folder = plot_folder
    ctx.obj.generated = generated
    ctx.obj.plot_extensions = extension

    flags = ctx.obj
    
    evt_start = flags.job_idx * flags.nevts if flags.job_idx >=0 else 0
    dataset_num = ctx.obj.config.get("DATASET_NUM", 2)

    bins = utils.XMLHandler(ctx.obj.config["PART_TYPE"], ctx.obj.config["BIN_FILE"])
    geom_conv = utils.GeomConverter(bins)

    generated, energies = LoadSamples(flags, ctx.obj.config, geom_conv)

    total_evts = energies.shape[0]

    data = []
    for dataset in ctx.obj.config["EVAL"]:
        with h5py.File(os.path.join(flags.data_folder, dataset), "r") as h5f:
            if flags.from_end:
                start = -int(total_evts)
                end = None
            else:
                start = evt_start
                end = start + total_evts
            show = h5f["showers"][start:end] / 1000.0
            if dataset_num <= 1:
                show = geom_conv.convert(geom_conv.reshape(show)).detach().numpy()
            data.append(show)

    data_dict = {
        "Geant4": np.reshape(data, ctx.obj.config["SHAPE"]),
        utils.name_translate(generated_file_path=ctx.obj.generated): generated,
    }

    plot_results(flags, ctx.obj.config, data_dict, energies)


def write_out(fout, flags, config, generated, energies, first_write = True, do_mask = False):

    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    dataset_num = config.get("DATASET_NUM", 2)

    if(not orig_shape): generated = generated.reshape(config["SHAPE_ORIG"])
    energies = np.reshape(energies,(energies.shape[0],-1))

    hgcal = config.get("HGCAL", False)
    shower_scale = config.get('SHOWERSCALE', 200.)

    if(do_mask  and dataset_num > 1):
        #mask for voxels that are always empty
        mask_file = os.path.join(flags.data_folder,config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
        if(not os.path.exists(mask_file)):
            print("Creating mask based on data batch")
            mask = np.sum(generated, 0)==0  # TODO ????

        else:
            with h5py.File(mask_file,"r") as h5f:
                mask = h5f['mask'][:]
        generated = generated *(np.reshape(mask,(1,-1))==0)

    generated = np.reshape(generated, config['SHAPE_ORIG'])
    if(first_write):
        print("Creating {}".format(fout))
        shape = list(config['SHAPE_ORIG'])
        shape[0] = None
        if(not hgcal):
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
            if(not hgcal):
                utils.append_h5(h5f, 'showers', (1./shower_scale) * generated)
                utils.append_h5(h5f, 'incident_energies', (1./shower_scale) * energies)
            else:
                utils.append_h5(h5f, 'showers', (1./shower_scale) * generated)
                utils.append_h5(h5f, 'gen_info', energies)

    first_write = True

    write_out(fout, flags, config, generated, energies, first_write=first_write)


def LoadSamples(flags, config, geom_conv):
    end = None if flags.nevts < 0 else flags.nevts
    with h5py.File(flags.generated, "r") as h5f:
        generated = h5f["showers"][:end] / 1000.0
        energies = h5f["incident_energies"][:end] / 1000.0
    energies = np.reshape(energies, (-1, 1))

    if config.get("DATASET_NUM", 2) <= 1:
        generated = geom_conv.convert(geom_conv.reshape(generated)).detach().numpy()
    generated = np.reshape(generated, config["SHAPE"])
    return generated, energies


def plot_results(flags, config, data_dict, energies): 
    plot_routines = {
        "Energy per layer": plots.ELayer(flags, config),
        "Energy": plots.HistEtot(flags, config),
        "2D Energy scatter split": plots.ScatterESplit(flags, config),
        "Energy Ratio split": plots.HistERatio(flags, config),
    }
    if not flags.layer_only:
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


def run_inference(flags, config, model):
    data_loader = utils.load_data(flags, config, eval=True)

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

    generated, energies = model.generate(data_loader, sample_steps, flags.debug, flags.sample_offset)
    if(flags.generated == ""):
        fout = f"{model_instance.checkpoint_folder}/generated_{config['CHECKPOINT_NAME']}_{flags.sample_algo}{sample_steps}.h5"
    else:
        fout = flags.generated

    write_out(fout, flags, config, generated, energies, first_write=True)

if __name__ == "__main__":
    inference()
