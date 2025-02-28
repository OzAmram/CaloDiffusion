from argparse import ArgumentParser
import os

import numpy as np
import h5py as h5

from calodiffusion.utils import utils
import calodiffusion.utils.plots as plots
from calodiffusion.utils.utils import LoadJson


def plot_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-folder", default="./data/", help="Folder containing data and MC files"
    )
    parser.add_argument(
        "--plot-folder", default="./plots", help="Folder to save results"
    )
    parser.add_argument(
        "--plot-reshape",
        default=False,
        action="store_true",
        help="Plots in embeded space",
    )
    parser.add_argument(
        "--geant-only",
        default=False,
        action="store_true",
        help="Plots only of geant distribution",
    )
    parser.add_argument(
        "--cms", default=False, action="store_true", help="CMS style for the plots"
    )
    parser.add_argument("--generated", "-g", default="", help="Generated showers")
    parser.add_argument(
        "--config", "-c", default="config_dataset2.json", help="Training parameters"
    )
    parser.add_argument(
        "--EMin", default=-1.0, type=float, help="Min cell energy threshold"
    )
    parser.add_argument(
        "-n", "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument("--plot-label", default="", help="Add to plot")

    parser.add_argument(
        "--layer-only",
        default=False,
        action="store_true",
        help="Only sample layer energies",
    )

    parser.add_argument(
        "--model",
        default="diffusion",
        help="Diffusion model to load.",
    )

    flags = parser.parse_args()
    config = LoadJson(flags.config)
    return flags, config


def LoadSamples(fname, flags, config, NN_embed=None, nevts=-1):

    dataset_num = config.get("DATASET_NUM", 2)
    shower_scale = config.get("SHOWERSCALE", 0.001)
    hgcal = config.get("HGCAL", False)
    end = None if nevts < 0 else nevts

    if (not hgcal) or flags.plot_reshape:
        shape_plot = config["SHAPE_FINAL"]
    else:
        shape_plot = config["SHAPE_PAD"]

    with h5.File(fname, "r") as h5f:
        if hgcal:
            generated = h5f["showers"][:end, :, : config["MAX_CELLS"]] * shower_scale
            energies = h5f["gen_info"][:end, 0]
            gen_info = np.array(h5f["gen_info"][:end, :])
        else:
            generated = h5f["showers"][:end] * shower_scale
            energies = h5f["incident_energies"][:end] * shower_scale

    energies = np.reshape(energies, (-1, 1))
    if flags.plot_reshape:
        if dataset_num <= 1:
            generated = NN_embed.convert(NN_embed.reshape(generated)).detach().numpy()
        elif hgcal:
            generated = torch.from_numpy(generated.astype(np.float32)).reshape(
                config["SHAPE_PAD"]
            )
            generated = NN_embed.enc(generated).detach().numpy()

        generated = np.reshape(generated, shape_plot)

    if flags.EMin > 0.0:
        mask = generated < flags.EMin
        generated = utils.apply_mask_conserveE(generated, mask)

    return generated, energies


def plot(flags, config):

    hgcal = config.get("HGCAL", False)
    data_dict = {}

    NN_embed = None
    if hgcal and flags.plot_reshape:
        NN_embed = HGCalConverter(bins=shape_embed, geom_file=config["BIN_FILE"])
        NN_embed.init()

    if not flags.geant_only:
        generated, energies = LoadSamples(
            flags.generated, flags, config, NN_embed, nevts=flags.nevts
        )
        print("Generated avg %.6f" % np.mean(generated))

        data_dict[utils.name_translate[flags.model]] = generated
        total_evts = energies.shape[0]

    data = []
    true_energies = []
    for dataset in config["EVAL"]:
        fname = os.path.join(flags.data_folder, dataset)
        showers, energies = LoadSamples(
            fname, flags, config, NN_embed, nevts=total_evts
        )
        data.append(showers)
        true_energies.append(energies)
        if data[-1].shape[0] >= total_evts:
            break

    geant_key = "Geant4"

    data_dict[geant_key] = np.concatenate(data)
    print("Geant avg %.6f" % np.mean(data_dict[geant_key]))

    if not os.path.exists(flags.plot_folder):
        os.system("mkdir %s" % flags.plot_folder)

    plot_routines = {
        "Energy per layer": plots.ELayer(flags, config),
        "Energy": plots.HistEtot(flags, config),
        "2D Energy scatter split": plots.ScatterESplit(flags, config),
        "Energy Ratio split": plots.HistERatio(flags, config),
        "Layer Sparsity": plots.SparsityLayer(flags, config),
    }

    # if(flags.geant_only):
    # plot_routines['IncidentE split'] = IncidentE

    if hgcal and not flags.plot_reshape:
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

    print("Saving plots to " + os.path.abspath(flags.plot_folder))

    for plotting_method in plot_routines.values():
        plotting_method(data_dict, energies)


if __name__ == "__main__":
    flags, config = plot_parser()
    plot(flags, config)
