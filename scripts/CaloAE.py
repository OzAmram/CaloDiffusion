import numpy as np
import time
import utils
import torch
import torch.nn as nn
import torch.nn.functional as f
from models import *
from torch.autograd import Variable
from torchinfo import summary



class CaloAE(nn.Module):
    """AE to encoder calorimeter data"""
    def __init__(self, data_shape,num_batch, config=None):
        super(CaloAE, self).__init__()
        self._data_shape = data_shape
        self._num_batch = num_batch

        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1

        if(self.config['ACT'].lower() == 'swish'): self.activation = nn.SiLU
        elif(self.config['ACT'].lower() == 'relu'): self.activation = nn.ReLU
        else: 
            print("Unrecognized activation fn : %s" % self.config['ACT'].lower())
            self.activation = nn.ReLU

        self.stride_size=self.config['STRIDE']
        self.kernel_size =self.config['KERNEL']
        self.layer_size = self.config['LAYER_SIZE_AE']
        self.nlayers = len(self.layer_size)
        self.dim_red = self.config['DIM_RED']

        total_dim_red = int(np.sum(self.dim_red))
        print("Dim red %i" % total_dim_red)
        total_dim_red = max(total_dim_red, 1)

        encoded_shape = list(self._data_shape)
        encoded_shape[-1] = int(self._data_shape[-1]/total_dim_red)
        encoded_shape[-2] = int(self._data_shape[-2]/total_dim_red)
        self.encoded_shape = encoded_shape

        self.encoder_model = self.BuildEncoder(self._data_shape)
        self.decoder_model = self.BuildDecoder(self._data_shape)

        print("data shape", self._data_shape)
        print("enc shape", self.encoded_shape)

        print("Encoder : \n")
        summary_shape = list(self._data_shape)
        summary_shape.insert(0, 1)
        #summary(self.encoder_model, summary_shape)

        print("Decoder : \n")
        summary_shape = list(self.encoded_shape)
        summary_shape.insert(0, 1)
        #summary(self.decoder_model, summary_shape)



    def BuildEncoder(self, input_shape):
        self.enc_layers = []
        in_channels = input_shape[0]
        for ilayer in range(self.nlayers):
            out_channels = self.layer_size[ilayer]
            lay = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size=self.kernel_size,padding='same', bias=True)
            self.enc_layers.append(lay)
            self.enc_layers.append(self.activation())
            in_channels = out_channels

            if(self.dim_red[ilayer] > 0):
                pool_lay = nn.AvgPool3d(kernel_size = (1, self.dim_red[ilayer], self.dim_red[ilayer]))
                self.enc_layers.append(pool_lay)

        final_lay = nn.Conv3d(in_channels = in_channels, out_channels = 1, kernel_size=self.kernel_size,padding='same', bias=True)
        self.enc_layers.append(final_lay)
        Encoder = nn.Sequential(*self.enc_layers)
        return Encoder

    def BuildDecoder(self, input_shape):
        self.dec_layers = []
        in_channels = 1
        for ilayer in range(self.nlayers -1, -1, -1):
            out_channels = self.layer_size[ilayer]
            lay =  nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size=self.kernel_size,padding='same', bias=True)
            self.dec_layers.append(lay)
            self.dec_layers.append(self.activation())
            in_channels = out_channels

            if(self.dim_red[ilayer] > 0):
                up = nn.Upsample(scale_factor = (1, self.dim_red[ilayer], self.dim_red[ilayer]))
                self.dec_layers.append(up)

            #x = layers.BatchNormalization()(x)

        final_lay = nn.Conv3d(in_channels = in_channels, out_channels = input_shape[0], kernel_size=self.kernel_size,padding='same', bias=True)
        self.dec_layers.append(final_lay)
        Decoder = nn.Sequential(*self.dec_layers)

        return Decoder

    def forward(self, inputs):
        out = self.encoder_model(inputs)
        out = self.decoder_model(out)
        return out
