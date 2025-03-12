import click
from calodiffusion.train.optimize import Optimize
from datetime import datetime
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.utils.utils import dotdict

@click.group()
@click.option('-c', '--config', help='Configuration file')
@click.option('-o', '--objectives', default=["COUNT", "FPD"], multiple=True, help='Objectives to optimize')
@click.option('--name', 'study_name', default=f"search_study_{datetime.timestamp(datetime.now())}", help='Name of the study')
@click.option('--n-trials', type=int, default=30, help='Number of trials to run')
@click.option('--data-folder', default='./data/', help='Folder containing data and MC files')
@click.option('--results-folder', default='./optuna_reports', help='Folder to save results')
@click.option('-n', '--nevts', type=int, default=-1, help='Number of events to load')
@click.option('--frac', type=float, default=0.85, help='Fraction of total events used for training')
@click.pass_context
def optimize(ctx, config, objectives, study_name, n_trials, data_folder, results_folder, nevts, frac):
    args = dotdict({
        'config': config,
        'objectives': objectives,
        'study_name': study_name,
        'n_trials': n_trials,
        'data_folder': data_folder,
        'results_folder': results_folder,
        'nevts': nevts,
        'frac': frac,
    })
    ctx.ensure_object(dotdict)
    ctx.obj['args'] = args


@optimize.group()
def training():
    pass 

@training.group()
@click.pass_context
def layer(ctx):
    Optimize(flags=ctx.obj['args'], trainer=TrainLayerModel, objectives=ctx.obj.objectives)()

@training.group()
@click.pass_context
def diffusion(ctx): 
    Optimize(flags=ctx.obj['args'], trainer=TrainDiffusion, objectives="")()

@optimize.group()
def inference():
    pass

@inference.group("layer")
@click.pass_context
def layer_inference(ctx):
    Optimize(flags=ctx.obj['args'], trainer=TrainLayerModel, objectives=ctx.obj.objectives)()

@inference.group()
@click.pass_context
def diffusion_inference(ctx): 
    Optimize(flags=ctx.obj['args'], trainer=TrainDiffusion, objectives="")()


if __name__ == "__main__":
    optimize()
