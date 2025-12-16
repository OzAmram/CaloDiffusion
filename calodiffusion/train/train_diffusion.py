import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from calodiffusion.utils import utils
from calodiffusion.train.train import Train
from calodiffusion.models.calodiffusion import CaloDiffusion


class TrainDiffusion(Train): 
    def __init__(self, flags, config, load_data=True, save_model:bool=True) -> None:
        super().__init__(flags, config, load_data=load_data, save_model=save_model)

    def init_model(self):
        self.model = CaloDiffusion(
            self.config, n_steps=self.config["NSTEPS"], loss_type=self.config['LOSS_TYPE']
        )
    
    def training_loop(self, optimizer, scheduler, early_stopper, start_epoch, num_epochs, training_losses, val_losses):
    
        tqdm = utils.import_tqdm()
        cold_diffu = self.config.get("COLD_DIFFU", False)
        cold_noise_scale = self.config.get("COLD_NOISE", 1.0)

        #fixed noise levels for  the validation loss for stability
        if(self.loader_val is not None):
            val_rnd = torch.randn( (len(self.loader_val)+1,self.batch_size,), device=self.device)
            print(val_rnd.shape)


        # training loop
        min_validation_loss = 99999.0
        epoch = start_epoch
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
                    data=data, energy=E, noise=noise, layers=layers, time=t
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
            if(self.loader_val is not None):
                for i, (vE, vlayers, vdata) in tqdm(
                    enumerate(self.loader_val, 0), unit="batch", total=len(self.loader_val)
                ):
                    #dumb fix
                    if(i >= val_rnd.shape[0]): break

                    vdata = vdata.to(device=self.device)
                    vE = vE.to(device=self.device)
                    vlayers = vlayers.to(device=self.device)


                    noise = torch.randn_like(vdata)

                    #use fixed time steps for stable val loss
                    rnd_normal = val_rnd[i].to(device=self.device)

                    #make sure shape of last batch handled properly
                    if(vE.shape[0] != self.batch_size):
                        rnd_normal = rnd_normal[:vE.shape[0]]

                    if cold_diffu:
                        noise = self.model.gen_cold_image(vE, cold_noise_scale, noise)

                    batch_loss = self.model.compute_loss(
                        vdata, vE, noise=noise, layers=vlayers, rnd_normal=rnd_normal,
                    )

                    val_loss += batch_loss.item()
                    del vdata, vE, vlayers, noise, batch_loss

                val_loss = val_loss / len(self.loader_val)
                val_losses[epoch] = val_loss
                print("val_loss: " + str(val_loss), flush=True)

            scheduler.step(torch.tensor([train_loss]))

            if val_loss < min_validation_loss:
                if self.save_model: 
                    torch.save(
                        self.model.state_dict(), os.path.join(self.checkpoint_folder, "best_val.pth")
                    )
                min_validation_loss = val_loss

            if early_stopper.early_stop(val_loss):
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
