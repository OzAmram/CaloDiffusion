<<<<<<< HEAD
import argparse
=======
>>>>>>> ebacf4f (layer model function with training and inference methods)
import os
import click 

import numpy as np
import h5py

from calodiffusion.utils import utils
<<<<<<< HEAD
from calodiffusion.train import Diffusion
#models = {model.__name__: model for model in [Diffusion]}
models = {'diffusion': Diffusion}


def inference_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data-folder", default="./data/", help="Folder containing data and MC files"
    )
    parser.add_argument(
        "--plot-folder", default="./plots", help="Folder to save results"
    )
    parser.add_argument(
        "--plot", action='store_true', help='generate plot at the same time as running inference.'
    )
    parser.add_argument(
        "--model-folder", dest='checkpoint_folder', default="../models/", help="Folder containing trained model"
    )
    parser.add_argument("--generated", "-g", default="", help="File name for generated showers")
    parser.add_argument("--model-loc", default="test", help="Location of model")
    parser.add_argument(
        "--config", "-c", default="config_dataset2.json", help="Training parameters"
    )
    parser.add_argument(
        "-n", "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for generation"
    )
    parser.add_argument(
        "--model",
        default="diffusion",
        help="Diffusion model to load.",
        choices=models.keys()
    )
    parser.add_argument("--plot-label", default="", help="Add to plot")

    parser.add_argument(
        "--sample", action="store_true", default=False, help="Sample from learned model"
    )
    parser.add_argument(
        "--reclean", action="store_true", default=False, help="Redo preprocessing on loaded sample"
    )
    parser.add_argument(
        "--sample-steps",
        default=None,
        type=int,
        help="How many steps for sampling (override config)",
    )
    parser.add_argument(
        "--sample-offset",
        default=0,
        type=int,
        help="Skip some iterations in the sampling (noisiest iters most unstable)",
    )
    
    parser.add_argument("--frac", help=argparse.SUPPRESS, default=0.75)
=======
import calodiffusion.utils.plots as plots
from calodiffusion.utils.utils import LoadJson

from calodiffusion.train import Diffusion, TrainLayerModel

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

@click.group()
@click.option("--config", required=True)
@click.option("--data-folder", default="./data/", help="Folder containing data and MC files")
@click.option("--checkpoint-folder", default="./trained_models/", help="Folder to save checkpoints")
@click.option("-n", "--n-events", default=-1, type=int, help="Number of events to load")
@click.option("--job-idx", default=-1, type=int, help="Split generation among different jobs")
@click.option("--layer-only/--no-layer", default=False, help="Only sample layer energies")
@click.option("--debug/--no-debug", default=False, help="Debugging options")
@click.pass_context
def inference_parser(ctx, debug, config, data_folder, checkpoint_folder, layer_only, job_idx, n_events): 
    ctx.ensure_object(dotdict)
    
    ctx.obj.config = LoadJson(config)
    ctx.obj.checkpoint_folder = checkpoint_folder
    ctx.obj.data_folder = data_folder
    ctx.obj.debug = debug
    ctx.obj.job_idx = job_idx
    ctx.obj.nevts = n_events 
    ctx.obj.layer_only = layer_only


@inference_parser.group()
@click.option("--sample-steps", default=400, type=int, help="How many steps for sampling (override config)")
@click.option("--sample-offset", default=0, type=int, help="Skip some iterations in the sampling (noisiest iters most unstable)")
@click.option("--sample-algo", default="ddpm", help="What sampling algorithm (ddpm, ddim, cold, cold2)")
@click.option("--model-loc", default=None, help="Specific folder for loading existing model", required=True)
@click.pass_context
def sample(ctx, sample_steps, sample_algo, sample_offset, model_loc):
    ctx.obj.model_loc = model_loc
    ctx.obj.sample_steps = sample_steps
    ctx.obj.sample_algo = sample_algo 
    ctx.obj.sample_offset = sample_offset

@sample.command()
@click.option("--layer-model", required=True)
@click.pass_context
def layer(ctx, layer_model): 
    ctx.obj.config['layer_model'] = layer_model
    inference(ctx.obj, ctx.obj.config, model=TrainLayerModel)

@sample.command()
@click.pass_context
def diffusion(ctx):
    inference(ctx.obj, ctx.obj.config, model=Diffusion)
>>>>>>> ebacf4f (layer model function with training and inference methods)


@inference_parser.command()
@click.option("-g", "--generated", help="Generated showers", required=True)
@click.option("--plot-label", default="", help="Labels for the plot")
@click.option("--plot-folder", default="./plots", help="Folder to save results")
@click.pass_context
def plot(ctx, generated, plot_label):
    ctx.obj.plot_label = plot_label

<<<<<<< HEAD
    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    config['flags'] = flags  # Access shortcut
    return flags, config
=======
    flags = ctx.obj
    evt_start = flags.job_idx * flags.nevts
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
        utils.name_translate.get(flags.model, flags.model): generated,
    }

    plot_results(flags, ctx.config, data_dict, energies)
>>>>>>> ebacf4f (layer model function with training and inference methods)


def write_out(fout, flags, config, generated, energies, first_write = True, do_mask = False):

    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    dataset_num = config.get("DATASET_NUM", 2)

    if(not orig_shape): generated = generated.reshape(config["SHAPE_ORIG"])
    energies = np.reshape(energies,(energies.shape[0],-1))

    hgcal = config.get("HGCAL", False)
    shower_scale = config.get('SHOWERSCALE', 200.)

<<<<<<< HEAD
    if(do_mask  and dataset_num > 1):
        #mask for voxels that are always empty
        mask_file = os.path.join(flags.data_folder,config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
        if(not os.path.exists(mask_file)):
            print("Creating mask based on data batch")
            mask = np.sum(generated, 0)==0  # TODO ????

        else:
            with h5py.File(mask_file,"r") as h5f:
                mask = h5f['mask'][:]
        generated = generated*(np.reshape(mask,(1,-1))==0)
=======
        if flags.debug: 
            data.append(d_batch)
>>>>>>> ebacf4f (layer model function with training and inference methods)

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

<<<<<<< HEAD
    write_out(fout, flags, config, generated, energies, first_write=first_write)


def inference(flags, config):
    data_loader, _ = utils.load_data(flags, config, eval=True)
=======
                fout_ex = "{}/{}_{}_norm_voxels_x0_step{}.{}".format(
                    flags.plot_folder,
                    config["CHECKPOINT_NAME"],
                    flags.model,
                    j,
                    ".png",
                )
                plot.Plot(flags, config)._histogram(
                    [x0s[j].cpu().reshape(-1), np.concatenate(data).reshape(-1)],
                    ["Diffu", "Geant4"],
                    ["blue", "black"],
                    xaxis_label="Normalized Voxel Energy",
                    num_bins=40,
                    normalize=True,
                    fname=fout_ex,
                )

            generated.append(gen)

    generated = np.concatenate(generated)
    energies = np.concatenate(energies)
    layers = np.concatenate(layers)

    if not orig_shape:
        generated = generated.reshape(config["SHAPE"])

    generated, energies = utils.ReverseNorm(
        generated,
        energies,
        layerE=layers,
        shape=config["SHAPE"],
        logE=config["logE"],
        binning_file=config["BIN_FILE"],
        max_deposit=config["MAXDEP"],
        emax=config["EMAX"],
        emin=config["EMIN"],
        showerMap=config["SHOWERMAP"],
        dataset_num=config.get("DATASET_NUM", 2),
        orig_shape=orig_shape,
        ecut=config["ECUT"],
    )

    return generated, energies


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


def inference(flags, config, model):
    data_loader = utils.load_data(flags, config, eval=True)
    dataset_num = config.get("DATASET_NUM", 2)
>>>>>>> ebacf4f (layer model function with training and inference methods)

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
<<<<<<< HEAD
    flags, config = inference_parser()
    inference(flags, config)
=======
    inference_parser()
>>>>>>> ebacf4f (layer model function with training and inference methods)
