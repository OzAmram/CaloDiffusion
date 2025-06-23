from argparse import ArgumentParser
import os

import numpy as np
import h5py as h5

from calodiffusion.utils.plots import RadialEnergyHGCal
from calodiffusion.utils.utils import *
from calodiffusion.utils.HGCal_utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-folder", default="./data/", help="Folder containing data and MC files"
    )
    parser.add_argument(
        "--plot-folder",
        default="./plots/hgcal_energy_test/",
        help="Folder to save results",
    )
    parser.add_argument(
        "--plot-reshape",
        default=False,
        action="store_true",
        help="Plots in embeded space",
    )
    parser.add_argument(
        "--cms", default=False, action="store_true", help="CMS style for the plots"
    )
    parser.add_argument("--generated", "-g", default="", help="Generated showers")
    parser.add_argument(
        "--config",
        "-c",
        default="calodiffusion/configs/config_HGCal.json",
        help="Training parameters",
    )
    parser.add_argument(
        "--EMin", default=-1.0, type=float, help="Min cell energy threshold"
    )
    parser.add_argument(
        "-n", "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument(
        "--layer-only", default=False, action='store_true', help="Only layers"
    )
    parser.add_argument("--plot-label", default="", help="Add to plot")

    flags = parser.parse_args()
    config = LoadJson(flags.config)
    return flags, config


def test_hgcal_energy_loader(flags, config):
    hgcal = True

    files = get_files(config["FILES"])[:3]

    batch_size = config["BATCH"]
    dataset_num = config.get("DATASET_NUM", 2)
    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    layer_norm = "layer" in config["SHOWERMAP"]
    geom_file = config.get("BIN_FILE", "")

    pre_embed = "pre-embed" in shower_embed
    shower_scale = config.get("SHOWERSCALE", 200.0)
    max_cells = config.get("MAX_CELLS", None)

    shape_orig = config.get("SHAPE_ORIG")
    shape_pad = config.get("SHAPE_PAD")
    shape_final = config.get("SHAPE_FINAL")

    geom_file = config.get("BIN_FILE")

    NN_embed = HGCalConverter(bins=shape_final, geom_file=geom_file, device=device).to(
        device=device
    )
    NN_embed.init()

    e_key = "gen_info"

    raw_shower = []
    data = []
    layers = []
    e = []

    for dataset in datasets:
        with h5.File(dataset, "r") as h5f:
            raw_shower_ = (
                h5f["showers"][0 : int(flags.nevts)].astype(np.float32) * shower_scale
            )
            raw_shower_ = raw_shower_[:, :, :max_cells]

            # process this dataset
            data_, e_, layers_ = DataLoader(
                dataset,
                shape=config["SHAPE_PAD"],
                emax=config["EMAX"],
                emin=config["EMIN"],
                hgcal=hgcal,
                nevts=flags.nevts,
                binning_file=geom_file,
                max_deposit=config[
                    "MAXDEP"
                ],  # noise can generate more deposited energy than generated
                logE=config["logE"],
                showerMap=config["SHOWERMAP"],
                shower_scale=shower_scale,
                max_cells=max_cells,
                dataset_num=dataset_num,
                orig_shape=orig_shape,
                config=config,
                embed=pre_embed,
                NN_embed=NN_embed,
            )
            e.append(e_)
            layers.append(layers_)
            if(not flags.layer_only):
                raw_shower.append(raw_shower_)
                data.append(data_)
            else:
                del raw_shower_, data_

    e = np.concatenate(e)
    layers = np.concatenate(layers)
        
    if(not flags.layer_only):
        data = np.concatenate(data)
        data = np.reshape(data, shape_pad)
        raw_shower = np.concatenate(raw_shower)

    print(layers.shape)
    print(shape_pad)

    if not pre_embed and not flags.layer_only:
        tdata = torch.tensor(data)

        data_enc = apply_in_batches(NN_embed.enc, tdata, device=device)
        data_dec = (
            apply_in_batches(NN_embed.dec, data_enc, device=device)
            .detach()
            .cpu()
            .numpy()
        )
        data_enc = data_enc.detach().cpu().numpy()

    else:
        data_enc = data_dec = data

    print("ShowerMap %s" % config["SHOWERMAP"])
    print("RAW: \n")
    # print(raw_shower[0,0,10])
    print("PREPROCESSED: \n")
    # print(data_[0,0,10])
    if layers is not None:
        totalE, layers_ = layers[:, 0], layers[:, 1:]
        print("TotalE MEAN %.4f, STD: %.5f" % (np.mean(totalE), np.std(totalE)))
        print("LAYERS MEAN %.4f, STD: %.5f" % (np.mean(layers_), np.std(layers_)))

    mean = np.mean(data_enc)
    std = np.std(data_enc)
    maxE = np.amax(data_enc)
    minE = np.amin(data_enc)

    maxEn = np.amax(e_)
    minEn = np.amin(e_)
    print("VOX MEAN: %.4f STD : %.5f" % (mean, std))
    print("MAX: %.4f MIN : %.5f" % (maxE, minE))
    print("maxE %.2f minE %.2f" % (maxEn, minEn))


    data_rev, e_rev = ReverseNorm(
        data_dec,
        e_,
        shape=shape_pad,
        layerE=layers,
        emax=config["EMAX"],
        emin=config["EMIN"],
        max_deposit=config[
            "MAXDEP"
        ],  # noise can generate more deposited energy than generated
        logE=config["logE"],
        showerMap=config["SHOWERMAP"],
        dataset_num=dataset_num,
        ecut=flags.EMin,
        orig_shape=orig_shape,
        hgcal=hgcal,
        embed=pre_embed,
        NN_embed=NN_embed,
    )

    data_rev[data_rev < flags.EMin] = 0.0

    print("REVERSED: \n")
    print("AVG DIFF: ", np.mean(data_rev[:100] - raw_shower[:100]))

    if not orig_shape:
        data_rev = np.reshape(data_rev, shape_orig)
        raw_shower = np.reshape(raw_shower, shape_orig)
        layer_rev = np.sum(data_rev, (2))
        raw_layer = np.sum(raw_shower, (2))
        print(data_rev.shape)
        print("AVG LAYER DIFF: ", np.mean(layer_rev[:100] - raw_layer[:100]))

    print("REVERSED: \n")
    # print(data_rev[0,0,10])
    print("AVG DIFF: ", torch.mean(torch.tensor(data_rev[:1000]) - raw_shower[:1000]))

    if False:
        data_dict = {}
        data_dict["Geant4"] = raw_shower
        data_dict["Embed- Pre-process - ReverseNorm - Decode"] = data_rev
        flags.model = "test"
        rad_plotter = RadialEnergyHGCal(flags, config)
        rad_plotter(data_dict, e_rev)

        layers = [3, 10, 24]
        geo = NN_embed.geom

        avg_shower_before = np.squeeze(np.mean(raw_shower, axis=0))
        avg_shower_after = np.squeeze(np.mean(data_rev, axis=0))

        avg_shower_ratio = avg_shower_after / avg_shower_before

        for ilay in layers:
            print(ilay)

            ncells = int(round(geo.ncells[ilay]))
            plot_shower_hex(
                geo.xmap[ilay][:ncells],
                geo.ymap[ilay][:ncells],
                avg_shower_before[ilay][:ncells],
                log_scale=False,
                nrings=geo.nrings,
                fout=flags.plot_folder + "avg_shower_lay%i_before.png" % (ilay),
            )
            plot_shower_hex(
                geo.xmap[ilay][:ncells],
                geo.ymap[ilay][:ncells],
                avg_shower_after[ilay][:ncells],
                log_scale=False,
                nrings=geo.nrings,
                fout=flags.plot_folder + "avg_shower_lay%i_after.png" % (ilay),
            )
            plot_shower_hex(
                geo.xmap[ilay][:ncells],
                geo.ymap[ilay][:ncells],
                avg_shower_ratio[ilay][:ncells],
                log_scale=False,
                nrings=geo.nrings,
                fout=flags.plot_folder + "avg_shower_lay%i_ratio.png" % (ilay),
            )


if __name__ == "__main__":
    flags, config = test_parser()
    test_hgcal_energy_loader(flags, config)
