import click
import os
from calodiffusion.train.optimize import Optimize
from datetime import datetime
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.utils import utils


@click.group()
@click.option('-c', '--config', help='Configuration file')
@click.option('-o', '--objectives', default=["COUNT", "FPD"], multiple=True, help='Objectives to optimize')
@click.option('--name', 'study_name', default=f"search_study_{datetime.timestamp(datetime.now())}", help='Name of the study')
@click.option('--n-trials', type=int, default=30, help='Number of trials to run')
@click.option('--data-folder', default='./data/', help='Folder containing data and MC files')
@click.option('--results-folder', default='./optuna_reports', help='Folder to save results')
@click.option('-n', '--nevts', type=int, default=-1, help='Number of events to load')
@click.option('--frac', type=float, default=0.85, help='Fraction of total events used for training')
@click.option('--hgcal/--no-hgcal', default=None, help='Use hgcal settings - overwrites config')
@click.pass_context
def optimize(ctx, config, objectives, study_name, n_trials, data_folder, results_folder, hgcal, nevts, frac):
    args = utils.dotdict({
        'objectives': objectives,
        'study_name': study_name,
        'n_trials': n_trials,
        'data_folder': data_folder,
        'results_folder': results_folder,
        'nevts': nevts,
        'frac': frac,
    })
    ctx.ensure_object(utils.dotdict)
    ctx.obj = args
    # Ensure the help menu still works
    if config is not None: 
        ctx.obj.config = utils.LoadJson(config)
        
        if hgcal is not None: 
            ctx.obj.config['HGCAL'] = hgcal
            ctx.obj.hgcal = hgcal
        else: 
            ctx.obj.hgcal = ctx.obj.config.get("HGCAL", False)

        os.environ["CONFIG"] = config 

@optimize.group()
def train():
    pass 

@train.command()
@click.pass_context
def layer(ctx):
    Optimize(flags=ctx.obj, trainer=TrainLayerModel, objectives=ctx.obj.objectives)()

@train.command()
@click.pass_context
def diffusion(ctx): 
    Optimize(flags=ctx.obj, trainer=TrainDiffusion, objectives=ctx.obj.objectives)()


@optimize.group()
@click.option("--model-loc", help='Path to the trained model weights')
@click.pass_context
def sample(ctx, model_loc):
    ctx.obj.model_loc = model_loc

@sample.command("layer")
@click.option("--layer-model", help='Path to trained layer model weights')
@click.pass_context
def layer_inference(ctx, layer_model):
    ctx.obj.config['layer_model'] = layer_model
    Optimize(flags=ctx.obj, trainer=TrainLayerModel, objectives=ctx.obj.objectives, inference=True)()

@sample.command("diffusion")
@click.pass_context
def diffusion_inference(ctx): 
    Optimize(flags=ctx.obj, trainer=TrainDiffusion, objectives=ctx.obj.objectives, inference=True)()


if __name__ == "__main__":
    optimize()
