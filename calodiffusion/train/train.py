from abc import ABC, abstractmethod
import os

import numpy as np
import torch


from calodiffusion.utils import utils
tqdm = utils.import_tqdm()

class Train(ABC): 
    def __init__(self, flags, config, load_data:bool=True, save_model: bool = True) -> None:
        self.device = utils.get_device()
        self.save_model = save_model

        if load_data: 
            self.loader_train, self.loader_val = utils.load_data(flags, config)
        
        self.config = config
        self.flags = flags
        self.batch_size = self.config.get("BATCH", 256)
        if self.save_model: 
            self.checkpoint_folder = f"{flags.checkpoint_folder.strip('/')}/{config['CHECKPOINT_NAME']}_{flags.model}/"
            if not os.path.exists(self.checkpoint_folder):
                os.makedirs(self.checkpoint_folder)

            os.system(
                "cp {} {}/config.json".format(flags.config, self.checkpoint_folder)
            )  # bkp of config file

    @abstractmethod
    def init_model(self): 
        raise NotImplementedError

    @abstractmethod
    def training_loop(
        self, 
        optimizer, 
        scheduler, 
        early_stopper, 
        start_epoch, 
        num_epochs, 
        training_losses, 
        val_losses): 

        raise NotImplementedError
    
    def pickup_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        early_stopper,
        n_epochs,
        restart_training,
    ):
        checkpoint_path = os.path.join(self.checkpoint_folder, "checkpoint.pth")

        if os.path.exists(checkpoint_path):
            print("Loading training checkpoint from %s" % checkpoint_path, flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            raise ValueError("No checkpoint at %s" % checkpoint_path)

        if "model_state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["model_state_dict"])
        elif len(checkpoint.keys()) > 1:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint.keys() and not restart_training:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint.keys() and not restart_training:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "early_stop_dict" in checkpoint.keys() and not restart_training:
            early_stopper.__dict__ = checkpoint["early_stop_dict"]

        training_losses = {}
        val_losses = {}
        start_epoch = 0

        if "train_loss_hist" in checkpoint.keys() and not restart_training:
            training_losses = checkpoint["train_loss_hist"]
            val_losses = checkpoint["val_loss_hist"]
            start_epoch = checkpoint["epoch"] + 1

        return model, optimizer, scheduler, start_epoch, training_losses, val_losses

    def save(
        self,
        model_state,
        epoch,
        name,
        training_losses,
        validation_losses,
        optimizer,
        scheduler,
        early_stopper,
    ):
        if self.save_model: 
            final_path = os.path.join(self.checkpoint_folder, f"{name}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss_hist": training_losses,
                    "val_loss_hist": validation_losses,
                    "early_stop_dict": early_stopper.__dict__,
                },
                final_path,
            )

        with open(self.checkpoint_folder + f"/{name}_training_losses.txt", "w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses.values()) + "\n")
        with open(self.checkpoint_folder + f"/{name}_validation_losses.txt", "w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in validation_losses.values()) + "\n")


    def train(self): 
        if not hasattr(self, "model"): 
            self.init_model()

        num_epochs = self.config.get("MAXEPOCH", 30)
        early_stopper = utils.EarlyStopper(
            patience=self.config["EARLYSTOP"], mode="diff", min_delta=1e-5
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["LR"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.1, patience=15, verbose=True
        )

        start_epoch = 0
        if self.flags.load:
            model, optimizer, scheduler, start_epoch, training_losses, val_losses = (
                self.pickup_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopper=early_stopper,
                    n_epochs=num_epochs,
                    restart_training=self.flags.reset_training,
                )
            )
        else: 
            training_losses = dict()
            val_losses = dict()

        model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper = self.training_loop(
            optimizer, 
            scheduler, 
            early_stopper, 
            start_epoch, 
            num_epochs, 
            training_losses, 
            val_losses
        )    
        # Also save at the end of training
        self.save(
            model.state_dict(),
            epoch=epoch,
            name="final",
            training_losses=training_losses,
            validation_losses=val_losses,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
        )
