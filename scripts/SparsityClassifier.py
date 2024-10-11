import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from sampling import *
from models import *


class SparsityClassifier(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape = None, config=None, R_Z_inputs = False, NN_embed = None, layer_model = None):
        super(SparsityClassifier, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.fully_connected = False
        self.NN_embed = NN_embed
        self.layer_model = layer_model


        if config is None:
            raise ValueError("Config file not given")
        
        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')


        cond_dim = config['COND_SIZE_UNET']
        layer_sizes = config['LAYER_SIZE_UNET']
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)

        #layer energies in conditional info or not
        self.layer_cond = False
        cond_size = 1
        if(config.get("HGCAL", False)): 
            cond_size += 2
        print("Cond size %i" % cond_size)

        self.loss = nn.BCELoss()




        if(self.fully_connected):
            #fully connected network architecture
            self.model = ResNet(cond_emb_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'], hidden_dim = 512)

            self.R_Z_inputs = False
            self.phi_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1,1], [1]]


        else:
            RZ_shape = config['SHAPE_FINAL'][1:]

            self.R_Z_inputs = config.get('R_Z_INPUT', False)
            self.phi_inputs = config.get('PHI_INPUT', False)

            in_channels = 1

            dataset_num = config['DATASET_NUM']
            self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = RZ_shape, dataset_num = dataset_num)
            self.phi_image = create_phi_image(device, shape = RZ_shape, dataset_num = dataset_num)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [[1,cond_size]], [1]]

            self.activation = nn.Sigmoid()


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = False, cond_size = cond_size, time_embed = False, no_time = True )


        print("\n\n Model: \n")
        summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d, strict = True):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new, strict = strict)


    def add_RZPhi(self, x):
        if(len(x.shape) < 3): return x
        cats = [x]
        const_shape = (x.shape[0], *((1,) * (len(x.shape) - 1)))
        if(self.R_Z_inputs):

            batch_R_image = self.R_image.repeat(const_shape).to(device=x.device)
            batch_Z_image = self.Z_image.repeat(const_shape).to(device=x.device)

            cats+= [batch_R_image, batch_Z_image]
        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat(const_shape).to(device=x.device)

            cats += [batch_phi_image]

        if(len(cats) > 1):
            return torch.cat(cats, axis = 1)
        else: 
            return x
    

    def compute_loss(self, shower_gen, shower_target, E, layers = None,):

        preds = self.pred(shower_gen, E=E, layers = layers)

        is_active = (shower_target > 0.).float()

        loss = self.loss(preds, is_active)

        return loss
  

    def pred(self, x, E, layers = None ):
        model = self.model

        if(self.NN_embed is not None ): x = self.NN_embed.enc(x).to(x.device)
        if(self.layer_cond and layers is not None): E = torch.cat([E, layers], dim = 1)

        out = model(self.add_RZPhi(x), cond=E, time=0. )

        if(self.NN_embed is not None ): out = self.NN_embed.dec(out).to(x.device)
        out = self.activation(out)
        return out


    def __call__(self, x, **kwargs):
        #sometimes want to call Unet directly, sometimes need wrappers
        if('cond' in kwargs.keys()):
            return self.model(x, **kwargs)
        else:
            return self.denoise(x, **kwargs)



