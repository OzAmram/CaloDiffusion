from calodiffusion.train.optimize import Optimize
from calodiffusion.train import Diffusion
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime

models = {model.__name__: model for model in [Diffusion]}

def argparse(): 
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", help="")
    parser.add_argument("-o", "--objectives", default=["COUNT", "FPD"])
    parser.add_argument("--name", dest='study_name', default=f"search_study_{datetime.timestamp(datetime.now())}")

    parser.add_argument(
        "--data-folder", default="./data/", help="Folder containing data and MC files"
    )
    parser.add_argument(
        "--plot-folder", default="./plots", help="Folder to save results"
    )

    parser.add_argument(
        "-n", "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.85,
        help="Fraction of total events used for training",
    )

    parser.add_argument(
        "--model",
        default="Diffusion",
        help="Diffusion model to load.",
        choices=models.keys()
    )

    parser.add_argument(
        "--load",
        default=False,
        type=bool, 
        help=SUPPRESS
    )
    return parser.parse_args()

if __name__ == "__main__": 
    args = argparse()
    Optimize(flags=args, trainer=models[args.model], objectives=args.objectives)()