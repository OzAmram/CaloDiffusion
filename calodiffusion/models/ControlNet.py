import torch
import torch.nn as nn

from utils import *
from models import *
from calodiffusion.models.CaloDiffu import *


class ControlledUNet(nn.Module):
    #Unet plus control net
    def __init__(self, UNet, ControlNet):
        super().__init__()
        self.UNet = UNet
        self.ControlNet = ControlNet
        self.nsteps = self.UNet.nsteps
        self.nets = nn.ModuleList([self.UNet, self.ControlNet])
        #Zero convolutions for downsampling and hidden state layers
        self.control_adds = nn.ModuleList([])
        for i in range(len(self.UNet.model.downs)):
            self.control_adds.append(ScalarAddLayer())

        #For middle hidden state
        self.control_adds.append(ScalarAddLayer())
        self.control_adds.append(ScalarAddLayer())

        
    def denoise(self, x, c_x = None, E =None, sigma=None, model = None, layers = None, layer_sample = False, controls = None):

        if(c_x is None):
            avg_showers, std_showers = self.UNet.lookup_avg_std_shower(E)
            c_x = avg_showers

        #Prepare controlnet inputs
        t_emb = self.UNet.do_time_embed(embed_type = self.UNet.time_embed, sigma = sigma.reshape(-1))
        conds = E
        if(self.ControlNet.layer_cond and layers is not None): conds = torch.cat([E, layers], dim = 1)
        if(self.ControlNet.NN_embed is not None): c_x = self.ControlNet.NN_embed.enc(c_x).to(c_x.device)

        control_hs = self.ControlNet.model.get_hiddens(self.ControlNet.add_RZPhi(c_x), conds, t_emb)

        controls = list(zip(self.control_adds, control_hs))

        out = self.UNet.denoise(x, conds, sigma, controls = controls)
        return out

    def __call__(self, x, **kwargs):
        return self.denoise(x, **kwargs)

    def Sample(self, E, layer_sample = False, model = None, **kwargs):

        if(layer_sample): 
            return self.UNet.Sample(E, layer_sample = True, model = model, **kwargs)

        else:
            return self.UNet.Sample(E, model = self, **kwargs)







