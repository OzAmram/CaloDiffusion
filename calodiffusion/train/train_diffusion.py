import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import h5py

from calodiffusion.utils import utils
from calodiffusion.utils.XMLHandler import XMLHandler
from calodiffusion.train.train import Train
from calodiffusion.models.CaloDiffu import CaloDiffu


class Diffusion(Train): 
    def __init__(self, flags, config, load_data=True) -> None:
        super().__init__(flags, config, load_data=load_data)

    def init_model(self):
        cold_diffu = self.config.get("COLD_DIFFU", False)

        training_obj = self.config.get("TRAINING_OBJ", "noise_pred")
        dataset_num = self.config.get("DATASET_NUM", 2)
        shower_embed = self.config.get("SHOWER_EMBED", "")
        orig_shape = "orig" in shower_embed

        avg_showers = std_showers = E_bins = None
        if cold_diffu:
            f_avg_shower = h5py.File(self.config["AVG_SHOWER_LOC"])
            # Already pre-processed
            avg_showers = torch.from_numpy(
                f_avg_shower["avg_showers"][()].astype(np.float32)
            ).to(device=self.device)
            std_showers = torch.from_numpy(
                f_avg_shower["std_showers"][()].astype(np.float32)
            ).to(device=self.device)
            E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(
                device=self.device
            )

        NN_embed = None
        if "NN" in shower_embed:
            if dataset_num == 1:
                bins = XMLHandler("photon", self.config["BIN_FILE"])
            else:
                bins = XMLHandler("pion", self.config["BIN_FILE"])

            NN_embed = utils.NNConverter(bins=bins).to(device=self.device)

        shape = self.config["SHAPE_PAD"][1:] if (not orig_shape) else self.config["SHAPE_ORIG"][1:]
        self.model = CaloDiffu(
            shape,
            config=self.config,
            training_obj=training_obj,
            NN_embed=NN_embed,
            nsteps=self.config["NSTEPS"],
            cold_diffu=cold_diffu,
            avg_showers=avg_showers,
            std_showers=std_showers,
            E_bins=E_bins,
            layer_model=None
        ).to(device=self.device)
    
    def training_loop(self, optimizer, scheduler, early_stopper, start_epoch, num_epochs, training_losses, val_losses):
    
        tqdm = utils.import_tqdm()
        cold_diffu = self.config.get("COLD_DIFFU", False)
        cold_noise_scale = self.config.get("COLD_NOISE", 1.0)
        loss_type = self.config.get("LOSS_TYPE", "l2")
        # training loop
        min_validation_loss = 99999.0
        for epoch in range(start_epoch, num_epochs):
            print("Beginning epoch %i" % epoch, flush=True)
            train_loss = 0

            self.model.train()
            for i, (E, layers, data) in tqdm(
                enumerate(self.loader_train, 0), unit="batch", total=len(self.loader_train)
            ):
                self.model.zero_grad()
                optimizer.zero_grad()

                data = data.to(device=self.device)
                E = E.to(device=self.device)
                layers = layers.to(device=self.device)

                t = torch.randint(0, self.model.nsteps, (data.size()[0],), device=self.device).long()
                noise = torch.randn_like(data)

                if cold_diffu:  # cold diffusion interpolates from avg showers instead of pure noise
                    noise = self.model.gen_cold_image(E, cold_noise_scale, noise)

                batch_loss = self.model.compute_loss(
                    data, E, noise=noise, layers=layers, t=t, loss_type=loss_type
                )
                batch_loss.backward()

                optimizer.step()
                train_loss += batch_loss.item()

                del data, E, layers, noise, batch_loss

            train_loss = train_loss / len(self.loader_train)
            training_losses[epoch] = train_loss

            print("loss: " + str(train_loss))

            val_loss = 0
            self.model.eval()
            for i, (vE, vlayers, vdata) in tqdm(
                enumerate(self.loader_val, 0), unit="batch", total=len(self.loader_val)
            ):
                vdata = vdata.to(device=self.device)
                vE = vE.to(device=self.device)
                vlayers = vlayers.to(device=self.device)

                t = torch.randint(0, self.model.nsteps, (vdata.size()[0],), device=self.device).long()
                noise = torch.randn_like(vdata)
                if cold_diffu:
                    noise = self.model.gen_cold_image(vE, cold_noise_scale, noise)

                batch_loss = self.model.compute_loss(
                    vdata, vE, noise=noise, layers=vlayers, t=t, loss_type=loss_type
                )

                val_loss += batch_loss.item()
                del vdata, vE, vlayers, noise, batch_loss

            val_loss = val_loss / len(self.loader_val)
            val_losses[epoch] = val_loss
            print("val_loss: " + str(val_loss), flush=True)

            scheduler.step(torch.tensor([train_loss]))

            if val_loss < min_validation_loss:
                torch.save(
                    self.model.state_dict(), os.path.join(self.checkpoint_folder, "best_val.pth")
                )
                min_validation_loss = val_loss

            if early_stopper.early_stop(val_loss - train_loss):
                print("Early stopping!")
                break

            # save the model for each checkpoint
            self.model.eval()
            print("SAVING")
            self.save(
                self.model.state_dict(),
                epoch=epoch,
                name="checkpoint",
                training_losses=training_losses,
                validation_losses=val_losses,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopper=early_stopper,
            )
            
        return self.model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper
