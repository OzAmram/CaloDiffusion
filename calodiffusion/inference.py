from argparse import ArgumentParser
import os

import numpy as np
import h5py

from calodiffusion.utils import utils
import calodiffusion.utils.plots as plots
from calodiffusion.utils.utils import LoadJson

from calodiffusion.train import Diffusion
models = {model.__name__: model for model in [Diffusion]}


def inference_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-folder", default="./data/", help="Folder containing data and MC files"
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
    parser.add_argument("--generated", "-g", default=None, help="Generated showers")
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
        default="Diffusion",
        help="Diffusion model to load.",
        choices=models.keys()
    )
    parser.add_argument("--plot-label", default="", help="Add to plot")

    parser.add_argument(
        "--sample", action="store_true", default=False, help="Sample from learned model"
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
    parser.add_argument(
        "--sample_algo",
        default="ddpm",
        help="What sampling algorithm (ddpm, ddim, cold, cold2)",
    )

    parser.add_argument(
        "--layer-only",
        default=False,
        action="store_true",
        help="Only sample layer energies",
    )

    parser.add_argument(
        "--job-idx", default=-1, type=int, help="Split generation among different jobs"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debugging options"
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        default=False,
        help="Use events from end of file (usually holdout set)",
    )

    flags = parser.parse_args()
    config = LoadJson(flags.config)
    return flags, config


def model_forward(flags, config, data_loader, model, sample_steps):
    device = utils.get_device()
    tqdm = utils.import_tqdm()

    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed

    generated = []
    data = []
    energies = []
    layers = []
    for E, layers_, d_batch in tqdm(data_loader):
        E = E.to(device=device)
        d_batch = d_batch.to(device=device)

        batch_generated = model.Sample(
            E,
            layers=layers_,
            num_steps=sample_steps,
            cold_noise_scale=config.get("COLD_NOISE", 1.0),
            sample_algo=flags.sample_algo,
            debug=flags.debug,
            sample_offset=flags.sample_offset,
        )


        if flags.debug: 
            data.append(d_batch)

        energies.append(E)
        generated.append(batch_generated)

        if "layer" in config["SHOWERMAP"]:
            layers.append(layers_)

        # Plot the histograms of normalized voxels for both the diffusion model and Geant4
        if flags.debug:
            gen, all_gen, x0s = batch_generated
            for j in [
                0,
                len(all_gen) // 4,
                len(all_gen) // 2,
                3 * len(all_gen) // 4,
                9 * len(all_gen) // 10,
                len(all_gen) - 10,
                len(all_gen) - 5,
                len(all_gen) - 1,
            ]:
                fout_ex = "{}/{}_{}_norm_voxels_gen_step{}.{}".format(
                    flags.plot_folder,
                    config["CHECKPOINT_NAME"],
                    flags.model,
                    j,
                    ".png",
                )
                plots.Plot(flags, config)._histogram(
                    [all_gen[j].cpu().reshape(-1), np.concatenate(data).reshape(-1)],
                    ["Diffu", "Geant4"],
                    ["blue", "black"],
                    xaxis_label="Normalized Voxel Energy",
                    num_bins=40,
                    normalize=True,
                    fname=fout_ex,
                )

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


def plot(flags, config, data_dict, energies): 
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


def inference(flags, config):
    data_loader = utils.load_data(flags, config, eval=True)
    dataset_num = config.get("DATASET_NUM", 2)

    model_instance = models[flags.model](flags, config, load_data=False)
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

    generated, energies = model_forward(flags, config, data_loader, model=model, sample_steps=sample_steps)
    if dataset_num > 1:
        # mask for voxels that are always empty
        mask_file = os.path.join(
            flags.data_folder, config["EVAL"][0].replace(".hdf5", "_mask.hdf5")
        )
        if not os.path.exists(mask_file):
            print("Creating mask based on data batch")
            mask = np.sum(generated, 0) == 0

        else:
            with h5py.File(mask_file, "r") as h5f:
                mask = h5f["mask"][:]

        generated = generated * (np.reshape(mask, (1, -1)) == 0)

    fout = f"{model_instance.checkpoint_folder}/generated_{config['CHECKPOINT_NAME']}_{flags.sample_algo}{sample_steps}.h5"

    print("Creating " + fout)
    with h5py.File(fout, "w") as h5f:
        h5f.create_dataset(
            "showers",
            data=1000 * np.reshape(generated, (generated.shape[0], -1)),
            compression="gzip",
        )
        h5f.create_dataset(
            "incident_energies", data=1000 * energies, compression="gzip"
        )
    return generated, energies


if __name__ == "__main__":
    flags, config = inference_parser()

    evt_start = flags.job_idx * flags.nevts
    dataset_num = config.get("DATASET_NUM", 2)

    bins = utils.XMLHandler(config["PART_TYPE"], config["BIN_FILE"])
    geom_conv = utils.GeomConverter(bins)

    if flags.sample: 
        generated, energies = inference(flags, config)

    else: 
        generated, energies = LoadSamples(flags, config, geom_conv)

    if flags.plot or (flags.generated is not None): 
        total_evts = energies.shape[0]

        data = []
        for dataset in config["EVAL"]:
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
            "Geant4": np.reshape(data, config["SHAPE"]),
            utils.name_translate.get(flags.model, flags.model): generated,
        }

        plot(flags, config, data_dict, energies)