import copy
from typing import Union
import torch
from calodiffusion.models.diffusion import Diffusion
from calodiffusion.models.models import ResNet, CondUnet
from calodiffusion.utils import utils
from calodiffusion.utils.HGCal_utils import HGCalConverter

class CaloDiffusion(Diffusion): 
    def __init__(self, config: Union[str, dict], n_steps: int = 400, loss_type: str = 'l2'):
        super().__init__(config, n_steps, loss_type)

        self.fully_connected = "FCN" in self.config.get("SHOWER_EMBED", "")
        self.time_embed = self.config.get("TIME_EMBED", "sin")
        self.dataset_num = self.config.get("DATASET_NUM", 2)

        self.R_image, self.Z_image = utils.create_R_Z_image(
            self.device, dataset_num = self.dataset_num, scaled=True, shape=self.config["SHAPE_FINAL"][1:]
        )
        self.phi_image = utils.create_phi_image(self.device, shape=self.config["SHAPE_FINAL"][1:])
        self.training_objective = self.config.get("TRAINING_OBJ", "noise_pred")
        self.layer_cond = "layer" in config.get("SHOWERMAP", "") 

    def init_model(self):

        self.pre_embed = "pre-embed" in self.config['SHOWER_EMBED']
        self.hgcal = self.config.get("HGCAL", False)
        self.fully_connected = "FCN" in self.config.get("SHOWER_EMBED", "")

        self.NN_embed = self.init_embedding_model()
        self.do_embed = self.NN_embed is not None and (not self.pre_embed)


        if self.fully_connected: 
            model = ResNet(
                cond_emb_dim=self.config["COND_SIZE_UNET"],
                dim_in=self.config["SHAPE_ORIG"][1],
                num_layers=self.config["NUM_LAYERS_LINEAR"],
                hidden_dim=512,
            ).to(device=self.device)

        else: 

            in_channels = 1
            if self.config.get("R_Z_INPUT", False):
                in_channels = 3

            if self.config.get("PHI_INPUT", False):
                in_channels += 1
            
            cond_size = 2 + self.config["SHAPE_FINAL"][2] if "layer" in self.config.get("SHOWERMAP", "") else 1
            calo_summary_shape = [1, in_channels] + list(copy.copy(self.config["SHAPE_FINAL"][1:]))

            #extra conditioning info for hgcal
            if(self.hgcal): cond_size +=2

            model = CondUnet(
                cond_dim=self.config["COND_SIZE_UNET"],
                out_dim=1,
                channels=in_channels,
                layer_sizes=self.config["LAYER_SIZE_UNET"],
                block_attn=self.config.get("BLOCK_ATTN", False),
                mid_attn= self.config.get("MID_ATTN", False),
                cylindrical=self.config.get("CYLINDRICAL", False),
                compress_Z=self.config.get("COMPRESS_Z", False),
                data_shape=calo_summary_shape,
                cond_embed=(self.config.get("COND_EMBED", "sin") == "sin"),
                cond_size=cond_size,
                time_embed=(self.config.get("TIME_EMBED", "sin") == "sin"),
            ).to(device=self.device)



        return model

    def noise_generation(self, shape):
        return super().noise_generation(shape)

    def forward(self, x, E, time, layers, layer_sample=False, controls=None):

        if (self.do_embed):
            x = self.NN_embed.enc(x).to(x.device)
        if self.layer_cond and layers is not None:
            E = torch.cat([E, layers], dim=1)

        rz_phi = self.add_RZPhi(x)
        out = self.model(rz_phi, cond=E, time=time.to(torch.float32), controls=controls)

        if (self.do_embed):
            out = self.NN_embed.dec(out).to(x.device)

        return out
    
    def init_embedding_model(self):

        dataset_num = self.config.get("DATASET_NUM", 2)
        shower_embed = self.config.get("SHOWER_EMBED", "")
        
        NN_embed = None
        if ("NN" in shower_embed and not self.hgcal):
            if dataset_num == 1:
                bins = utils.XMLHandler("photon", self.config["BIN_FILE"])
            else:
                bins = utils.XMLHandler("pion", self.config["BIN_FILE"])

            NN_embed = utils.NNConverter(bins=bins).to(device=self.device)

        elif(self.hgcal and not self.pre_embed):
            trainable = self.config.get('TRAINABLE_EMBED', False)
            NN_embed = HGCalConverter(bins = self.config['SHAPE_FINAL'], geom_file = self.config['BIN_FILE'], device = self.device, trainable = trainable).to(device = self.device)
            if(not trainable): NN_embed.init(norm = self.pre_embed, dataset_num = dataset_num)

        return NN_embed

    def add_RZPhi(self, x):

        if len(x.shape) < 3:
            return x
        cats = [x]
        const_shape = (x.shape[0], *((1,) * (len(x.shape) - 1)))

        if not self.fully_connected and self.config.get("R_Z_INPUT", False): 
            batch_R_image = self.R_image.repeat(const_shape).to(device=self.device)
            batch_Z_image = self.Z_image.repeat(const_shape).to(device=self.device)

            cats += [batch_R_image, batch_Z_image]

        if not self.fully_connected and self.config.get("PHI_INPUT", False):
            batch_phi_image = self.phi_image.repeat(const_shape).to(device=self.device)

            cats += [batch_phi_image]

        if len(cats) > 1:
            return torch.cat(cats, axis=1)
        else:
            return x

    def do_time_embed(
        self,
        sigma=None,
    ):
        embed: dict[str, callable] = {
            "sigma": lambda: sigma / (1 + sigma**2).sqrt(), 
            "log": lambda:  0.5 * torch.log(sigma)
        }
        return embed[self.time_embed]()
    
    def denoise(self, x, E=None, sigma=None, layers = None, layer_sample=False, controls=None):
        t_emb = self.do_time_embed(sigma = sigma.reshape(-1))
        loss_function_name = type(self.loss_function).__name__

        c_skip, c_out, c_in = self.loss_function.get_scaling(sigma)


        pred = self.forward(x * c_in, E, t_emb, layers = layers, layer_sample=layer_sample)

        if('noise_pred' in loss_function_name):
            return (x - sigma * pred)

        elif('mean_pred' in loss_function_name):
            return pred
        elif ('hybrid' or 'minsnr') in loss_function_name:
            return (c_skip * x + c_out * pred)
        else:
            raise ValueError("??? Training obj %s" % loss_function_name)


    def __call__(self, x, **kwargs):
        return self.denoise(x, **kwargs)
