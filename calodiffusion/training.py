import click
from calodiffusion.utils import utils
from calodiffusion.train import Diffusion, TrainLayerModel

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@click.group()
@click.option(
    "-d", "--data-folder", default="../data/", help="Folder containing data and MC files"
)

@click.option(
    "-c",
    "--config",
    default="configs/test.json",
    help="Config file with training parameters",
)
@click.option(
    "--checkpoint",
    "checkpoint_folder",
    default="../models",
    help="Folder with checkpoints",
)
@click.option(
    "-n", "--nevts", type=int, default=-1, help="Number of events to load"
)
@click.option(
    "--frac",
    type=float,
    default=0.85,
    help="Fraction of total events used for training",
)
@click.option(
    "--load",
    is_flag=True,
    default=False,
    help="Load pretrained weights to continue the training",
)
@click.option("--seed", type=int, default=1234, help="Pytorch seed")
@click.option('--reclean/--no-reclean', default=False, help='Reclean data')
@click.option(
    "--reset_training", is_flag=True, default=False, help="Retrain"
)
@click.pass_context
def train(ctx, config, data_folder, checkpoint_folder, nevts, frac, load, seed, reclean, reset_training): 
    ctx.ensure_object(dotdict)

    ctx.obj.config = utils.LoadJson(config)

    ctx.obj.data_folder = data_folder  
    ctx.obj.checkpoint_folder = checkpoint_folder
    ctx.obj.nevts = nevts
    ctx.obj.frac = frac
    ctx.obj.load = load
    ctx.obj.seed = seed
    ctx.obj.reclean = reclean
    ctx.obj.reset_training = reset_training


@train.command()
@click.pass_context
def diffusion(ctx): 
    Diffusion(ctx.obj, ctx.obj.config).train()

@train.command()
@click.pass_context
def layer(ctx, ): 

    #self.layer_steps = self.config.get("LAYER_STEPS")
    #sampler_algo = self.config.get("LAYER_SAMPLER", "DDim")

    TrainLayerModel(ctx.obj, ctx.obj.config).train()


if __name__ == "__main__": 
    train()
