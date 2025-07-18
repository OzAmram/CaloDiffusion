import copy
from typing import Union
import torch
from calodiffusion.models.diffusion import Diffusion
from calodiffusion.models.models import ResNet, CondUnet
from calodiffusion.utils import utils
import calodiffusion.utils.HGCal_utils as hgcal_utils

class CaloDiffusion(Diffusion): 
    def __init__(self, config: Union[str, dict], n_steps: int = 400, loss_type: str = 'l2'):
        super().__init__(config, n_steps, loss_type)
        self.pre_embed = "pre-embed" in self.config['SHOWER_EMBED']
        self.hgcal = self.config.get("HGCAL", False)

        self.fully_connected = "FCN" in self.config.get("SHOWER_EMBED", "")
        self.time_embed = self.config.get("TIME_EMBED", "sin")
        self.dataset_num = self.config.get("DATASET_NUM", 2)

        self.R_image, self.Z_image = utils.create_R_Z_image(
            self.device, dataset_num = self.dataset_num, scaled=True, shape=self.config["SHAPE_FINAL"][1:]
        )
        self.phi_image = utils.create_phi_image(self.device, shape=self.config["SHAPE_FINAL"][1:])
        self.training_objective = self.config.get("TRAINING_OBJ", "noise_pred")
        self.layer_cond = "layer" in config.get("SHOWERMAP", "")

        self.model = self.init_model()
        self.NN_embed = self.init_embedding_model()
        self.do_embed = self.NN_embed is not None and (not self.pre_embed)


    def load_state_dict(self, state_dict, strict = True):
        base_model_name = list(state_dict.keys())[10].split('.')[0]
        if base_model_name!="model": 
            state_dict = {
                key.removeprefix(f"{base_model_name}."): value for key, value in state_dict.items() if key.split('.')[0] == base_model_name
            }
        return super().load_state_dict(state_dict, strict)
    
    def init_model(self):

        self.fully_connected = "FCN" in self.config.get("SHOWER_EMBED", "")

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
                resnet_block_groups=self.config.get("BLOCK_GROUPS", 8), 
                data_shape=calo_summary_shape,
                cond_embed=(self.config.get("COND_EMBED", "sin") == "sin"),
                cond_size=cond_size,
                time_embed=(self.config.get("TIME_EMBED", "sin") == "sin"),
            ).to(device=self.device)

        return model.to(self.device)

    def noise_generation(self, shape):
        return super().noise_generation(shape)

    def forward(self, x, E, time, layers, controls=None):
        if (self.do_embed):
            x = self.NN_embed.enc(x.to(torch.float32)).to(x.device)
        if (self.layer_cond) and (layers is not None):
            E = torch.cat([E, layers], dim=1)

        rz_phi = self.add_RZPhi(x).float()
        out = self.model(rz_phi, cond=E.float(), time=time.float(), controls=controls)

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
            NN_embed = hgcal_utils.HGCalConverter(bins = self.config['SHAPE_FINAL'], geom_file = self.config['BIN_FILE'], device = self.device, trainable = trainable).to(device = self.device)
            if not trainable: 
                NN_embed.init(norm = self.pre_embed, dataset_num = dataset_num)

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
            "sigma": lambda sigma: sigma / (1 + sigma**2).sqrt(), 
            "log": lambda sigma:  0.5 * torch.log(sigma)
        }
        return embed[self.time_embed](sigma)
    
    def denoise(self, x, E=None, sigma=None, layers = None, controls=None, do_time_embed=True):
        if do_time_embed:
            t_emb = self.do_time_embed(sigma = sigma.reshape(-1)).to(float)
        else:
            t_emb = sigma.reshape(sigma.shape[0], -1).to(float)

        loss_function_name = type(self.loss_function).__name__

        scales = self.loss_function.get_scaling(sigma)
        pred = self.forward(x * scales['c_in'], E, t_emb, layers = layers )

        if('noise_pred' in loss_function_name):
            return (x - sigma * pred)
        elif ('hybrid' or 'minsnr') in loss_function_name:
            return (scales['c_skip'] * x + scales['c_out'] * pred)
        else: 
            return pred


    def __call__(self, x, **kwargs):
        return self.denoise(x, **kwargs)
