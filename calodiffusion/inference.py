from argparse import ArgumentParser
import os

import numpy as np
import h5py

from calodiffusion.utils import utils
from calodiffusion.train import Diffusion
#models = {model.__name__: model for model in [Diffusion]}
models = {'diffusion': Diffusion}


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
    config = utils.LoadJson(flags.config)
    return flags, config


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
        generated = generated*(np.reshape(mask,(1,-1))==0)

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


def inference(flags, config):
    data_loader, _ = utils.load_data(flags, config, eval=True)

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

    generated, energies = model.generate(data_loader, sample_steps, flags.debug, flags.sample_offset)
    if(flags.generated == ""):
        fout = f"{model_instance.checkpoint_folder}/generated_{config['CHECKPOINT_NAME']}_{flags.sample_algo}{sample_steps}.h5"
    else:
        fout = flags.generated

    write_out(fout, flags, config, generated, energies, first_write=True)

if __name__ == "__main__":
    flags, config = inference_parser()
    inference(flags, config)