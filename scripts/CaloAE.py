
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import tensorflow.keras.backend as K
#import horovod.tensorflow.keras as hvd
import utils

class CaloAE(nn.Module):
    """AE to encoder calorimeter data"""
    def __init__(self, data_shape,num_batch, config=None):
        super(CaloAE, self).__init__()
        self._data_shape = data_shape
        self._num_batch = num_batch

        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.activation = self.config['ACT']

        self.stride_size=self.config['STRIDE']
        self.kernel_size =self.config['KERNEL']
        self.layer_size = self.config['LAYER_SIZE_AE']
        self.nlayers = len(self.layer_size)
        self.dim_red = self.config['DIM_RED']

        total_dim_red = int(np.sum(self.dim_red))
        print("Dim red %i" % total_dim_red)

        encoded_shape = list(self._data_shape)
        encoded_shape[-2] = int(self._data_shape[-2]/total_dim_red)
        encoded_shape[-3] = int(self._data_shape[-3]/total_dim_red)
        self.encoded_shape = encoded_shape

        self.encoder_model = self.BuildEncoder(self._data_shape)
        self.decoder_model = self.BuildEncoder(self._data_shape)
        inputs = Input((self._data_shape))
        #x = layers.Conv3D(1,kernel_size=self.kernel_size,padding='same', strides=1,use_bias=False,activation=self.activation)(inputs)

        z = self.Encoder(inputs)
        x = self.Decoder(z)

        self.model = keras.Model(inputs, x)
        
        self.encoder_model = keras.Model(inputs, z)

        enc_inputs = Input((encoded_shape))
        dec = enc_inputs
        for lay in self.dec_layers:
            dec = lay(dec)
        self.decoder_model = keras.Model(enc_inputs, dec)

        print(self.model.summary())


    def BuildEncoder(self, input_shape):
        self.enc_layers = []
        in_channels = input_shape[1]
        for ilayer in range(self.nlayers):
            out_channels = self.layer_size[ilayer]
            lay = nn.Conv3D(in_channels = in_channels, out_channels = out_channels, kernel_size=self.kernel_size,padding='same', bias=True)
            self.enc_layers.append(lay)
            self.enc_layers.append(self.activation)
            in_channels = out_channels

            if(self.dim_red[ilayer] > 0):
                pool_lay = nn.AvgPool3D(kernel_size = (1, self.dim_red[ilayer], self.dim_red[ilayer]))
                self.enc_layers.append(pool_lay)

        final_lay = nn.Conv3D(in_channels = in_channels, out_channels = 1, ,kernel_size=self.kernel_size,padding='same', strides=1, bias=True)
        self.enc_layers.append(final_lay)
        Encoder = nn.Sequential(*self.enc_layers)
        return Encoder

    def Decoder(self, input_shape):
        self.dec_layers = []
        in_channels = 1
        for ilayer in range(self.nlayers -1, 0, -1):
            out_channels = self.layer_size[ilayer]
            lay =  nn.Conv3D(in_channels = in_channels, out_channels = out_channels, kernel_size=self.kernel_size,padding='same', bias=True)
            self.dec_layers.append(lay)
            self.enc_layers.append(self.activation)
            in_channels = out_channels

            if(self.dim_red[ilayer] > 0):
                up = nn.UpSampling3D(kernel_size = (1, self.dim_red[ilayer], self.dim_red[ilayer]))
                self.dec_layers.append(up)

            #x = layers.BatchNormalization()(x)

        final_lay = nn.Conv3D(in_channels = in_channels, out_channels = input_shape[1], kernel_size=self.kernel_size,padding='same', bias=True)
        self.dec_layers.append(final_lay)
        Decoder = nn.Sequential(*self.enc_layers)

        return Decoder


    def AEModel(self):


        cnn_encoder = ConvBlocks(inputs,conv_sizes,stride_size=stride_size,kernel_size = kernel_size,nlayers = nlayers)
        cnn_decoder = ConvTransBlocks(cnn_encoder,conv_sizes,stride_size=stride_size,kernel_size = kernel_size)
        if len(self._data_shape) == 2:
            outputs = layers.Conv1D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)
        else:
            outputs = layers.Conv3D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)

        
        return inputs, outputs

            
    @tf.function
    def call(self,x):        
        return self.model(x)
