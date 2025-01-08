from argparse import ArgumentParser
from calodiffusion.utils import utils
from calodiffusion.train import Train, Diffusion


models: dict[str, Train] = {
    "diffusion": Diffusion, 
}

def training_settings():
    parser = ArgumentParser()

    parser.add_argument(
        "-d", "--data-folder", default="../data/", help="Folder containing data and MC files"
    )
    parser.add_argument(
        "--model",
        default="Diffu",
        help="Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE",
        choices=models.keys()
    )

    parser.add_argument(
        "-c",
        "--config",
        default="configs/test.json",
        help="Config file with training parameters",
    )
    parser.add_argument(
        "--checkpoint",
        dest='checkpoint_folder',
        default="../models",
        help="Folder with checkpoints",
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
        "--load",
        action="store_true",
        default=False,
        help="Load pretrained weights to continue the training",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Pytorch seed")
    parser.add_argument(
        "--reset_training", action="store_true", default=False, help="Retrain"
    )
    args = parser.parse_args()

    dataset_config = utils.LoadJson(args.config)
    return args, dataset_config

def train(): 
    args, config = training_settings()

    train_method = args.model
    models[train_method](args, config).train()


if __name__ == "__main__": 
    train()