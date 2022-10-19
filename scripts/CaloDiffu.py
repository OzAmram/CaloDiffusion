import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import tensorflow.keras.backend as K
#import horovod.tensorflow.keras as hvd
import utils


# tf and friends
tf.random.set_seed(1234)

class CaloDiffu(keras.Model):
    """Diffusion based generative model"""
    def __init__(self, data_shape,num_cond,num_batch,name='SGM',config=None):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self._num_cond = num_cond
        self._num_batch = num_batch
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = self.config['NSTEPS']


        if config is None:
            raise ValueError("Config file not given")
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        self.verbose = 1

        

        #Convolutional model for 3D images and dense for flatten inputs

        #nested arrays to get shapes right
        self.beta_start = [[[0.0001]]]
        self.beta_end = [[[0.02]]]

        self.betas = tf.linspace(self.beta_start, self.beta_end, self.nsteps, axis = 0)
        self.alphas = 1. - self.betas


        self.alphas_cumprod = tf.math.cumprod(self.alphas)

        self.sqrt_recip_alphas = tf.math.sqrt(1.0 / self.alphas)


        self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1. - self.alphas_cumprod)

        #shift all elements over by inserting unit value in first place
        alphas_cumprod_prev = tf.cast(np.insert(self.alphas.numpy(), 0, 1., axis = 0)[:-1], tf.float32)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        #print(self.betas[:10])
        #print(self.alphas[:10])
        #print(self.alphas_cumprod[:10])
        #print(alphas_cumprod_prev[:10])


            
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.activation = self.config['ACT']

        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self._num_cond))
        time_projection = inputs_time*self.projection*2*np.pi
        cond_projection = inputs_cond*self.projection*2*np.pi
        time_embed = tf.concat([tf.math.sin(time_projection),tf.math.cos(time_projection)],-1)
        cond_embed = tf.concat([tf.math.sin(cond_projection),tf.math.cos(cond_projection)],-1)
        
        time_embed = layers.Dense(self._num_embed,activation=self.activation)(time_embed)
        cond_embed = layers.Dense(self._num_embed,activation=self.activation)(cond_embed)
        time_embed = tf.concat([time_embed,cond_embed],-1)
        
        time_embed = layers.Dense(self._num_embed,activation=self.activation)(time_embed)
        time_embed = layers.Dense(self._num_embed,activation=self.activation)(time_embed)

        
        

        if len(self._data_shape) == 2:
            self.shape = (-1,1,1)
            inputs,outputs = self.ConvModel(time_embed)
        else:
            self.shape = (-1,1,1,1,1)
            inputs,outputs = self.ConvModel(time_embed)

        
        self.model = keras.Model(inputs=[inputs,inputs_time,inputs_cond],outputs=outputs)
        

        if self.verbose:
            print(self.model.summary())

            

    def ConvModel(self,time_embed):     
        inputs = Input((self._data_shape))
        conv_sizes = self.config['LAYER_SIZE']
        self.n_skips = 0

        def ConvBlocks(layer,conv_sizes,stride_size,kernel_size,nlayers):
            skip_layers = []
            
            layer_encoded = self.time_conv(layer,time_embed,conv_sizes[0],
                                           kernel_size=kernel_size,
                                           stride=1,padding='same')
            skip_layers.append(layer_encoded)
            for ilayer in range(1,nlayers):
                layer_encoded = self.time_conv(skip_layers[-1],time_embed,conv_sizes[ilayer],
                                               kernel_size=kernel_size,padding='same',
                                               #stride=stride_size,
                                               stride=1,
                                               )
                
                if(layer_encoded.shape[-2] >= stride_size[-2] and layer_encoded.shape[-2] % 2 == 0):
                    if len(self._data_shape) == 2:
                        layer_encoded = layers.AveragePooling1D(stride_size)(layer_encoded)
                    else:
                        layer_encoded = layers.AveragePooling3D(stride_size)(layer_encoded)
                else:
                    self.n_skips +=1

                skip_layers.append(layer_encoded)

            return skip_layers[::-1]

        def ConvTransBlocks(skip_layers,conv_sizes,stride_size,kernel_size):
            layer_decoded = self.time_conv(skip_layers[0],
                                           time_embed,conv_sizes[len(skip_layers)-1],
                                           stride = 1,
                                           kernel_size=kernel_size,padding='same')
            for ilayer in range(len(skip_layers)-1):
                layer_decoded = self.time_conv(layer_decoded,
                                               time_embed,conv_sizes[len(skip_layers)-2-ilayer],
                                               stride = 1,
                                               kernel_size=kernel_size,padding='same')
                if(ilayer >= self.n_skips):
                    if len(self._data_shape) == 2:
                        layer_decoded = layers.UpSampling1D(stride_size)(layer_decoded)
                        layer_decoded =layers.Conv1D(conv_sizes[len(skip_layers)-2-ilayer],
                                                     kernel_size=kernel_size,padding="same",
                                                     strides=1,use_bias=True,
                                                     activation=self.activation)(layer_decoded)
                    else:
                        layer_decoded = layers.UpSampling3D(stride_size)(layer_decoded)
                        layer_decoded =layers.Conv3D(conv_sizes[len(skip_layers)-2-ilayer],
                                                     kernel_size=kernel_size,padding="same",
                                                     strides=1,use_bias=True,
                                                     activation=self.activation)(layer_decoded)
                        
                    
                    layer_decoded = (layer_decoded+ skip_layers[ilayer+1])/np.sqrt(2)
                if len(self._data_shape) == 2:
                    layer_decoded =layers.Conv1D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=1,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=self.activation)(layer_decoded)
                else:
                    layer_decoded =layers.Conv3D(conv_sizes[len(skip_layers)-2-ilayer],
                                                 kernel_size=1,padding="same",
                                                 strides=1,use_bias=True,
                                                 activation=self.activation)(layer_decoded)
            layer_decoded = self.time_conv(layer_decoded,
                                           time_embed,conv_sizes[0],
                                           stride = 1,
                                           kernel_size=kernel_size,padding='same')
            return layer_decoded

        
        stride_size=self.config['STRIDE']
        kernel_size =self.config['KERNEL']
        nlayers =self.config['NLAYERS']


        cnn_encoder = ConvBlocks(inputs,conv_sizes,stride_size=stride_size,kernel_size = kernel_size,nlayers = nlayers)
        cnn_decoder = ConvTransBlocks(cnn_encoder,conv_sizes,stride_size=stride_size,kernel_size = kernel_size)
        if len(self._data_shape) == 2:
            outputs = layers.Conv1D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)
        else:
            outputs = layers.Conv3D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(cnn_decoder)

        
        return inputs, outputs
        

        
    def time_conv(self,input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True):
        ## Incorporate information from conditional inputs
        time_layer = layers.Dense(hidden_size,activation=self.activation,use_bias=False)(embed)

        if len(self._data_shape) == 2:
            layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                                  strides=1,use_bias=False,activation=self.activation)(input_layer)
            time_layer = tf.reshape(time_layer,(-1,1,hidden_size))
            layer=layer+time_layer
            # layer = self.activate(layer)
            layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,
                                  padding=padding,
                                  strides=1,use_bias=True)(layer) 
            
        else:
            layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                                  strides=1,use_bias=False,activation=self.activation)(input_layer)
            time_layer = tf.reshape(time_layer,(-1,1,1,1,hidden_size))
            layer=layer+time_layer
            layer = layers.Conv3D(hidden_size,kernel_size=1,
                                  padding=padding,
                                  strides=1,use_bias=True)(layer)

        layer = layers.BatchNormalization()(layer)
        layer = layers.Dropout(0.1)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer
    
    def activate(self,layer):
        if self.activation == 'leaky_relu':                
            return keras.layers.LeakyReLU(-0.01)(layer)
        elif self.activation == 'relu':
            return keras.activations.relu(layer)
        elif self.activation == 'swish':
            return keras.activations.swish(layer)
        else:
            raise ValueError("Activation function not supported!")   


        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        return tf.constant(tf.random.normal(shape=(1,self._num_embed//2),seed=100))*scale
        



    @tf.function
    def noise_image(self, data, random_t, noise = None):
        if(noise is None): noise = tf.random.normal((tf.shape(data)))
        sqrt_alphas_cumprod_t = tf.gather(self.sqrt_alphas_cumprod, random_t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, random_t)

        out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise

        return out




    @tf.function
    def train_step(self, inputs):
        eps=1e-5

        data,cond = inputs
        init_shape = tf.shape(data)

        random_t = tf.random.uniform(shape = [self._num_batch, 1], minval = 1, maxval = self.nsteps, dtype=tf.dtypes.int32)
        noise = tf.random.normal((tf.shape(data)))

        perturbed_x = self.noise_image(data, random_t, noise = noise)

        with tf.GradientTape() as tape:

            
            pred_noise = self.model([perturbed_x, random_t, cond])
            loss = tf.losses.mean_squared_error(noise, pred_noise)

        g = tape.gradient(loss, self.model.trainable_variables)
        g = [tf.clip_by_norm(grad, 1) for grad in g]

        self.optimizer.apply_gradients(zip(g, self.trainable_variables))        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, inputs):
        eps=1e-5
        data,cond = inputs

        random_t = tf.random.uniform([self._num_batch, 1], minval = 1, maxval = self.nsteps, dtype=tf.dtypes.int32)
        noise = tf.random.normal((tf.shape(data)))
        
        perturbed_x = self.noise_image(data, random_t, noise = noise)
        pred_noise = self.model([perturbed_x, random_t, cond])
        loss = tf.losses.mean_squared_error(noise, pred_noise)
        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

            
    @tf.function
    def call(self,x):        
        return self.model(x)

    def p_sample(self, x, cond, t, batch_size = 256):
        #reverse the diffusion process

        betas_t = tf.gather(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = tf.gather(self.sqrt_recip_alphas, t)

        #noise_pred = self.model.predict([x, t, cond], batch_size = batch_size)
        noise_pred = self.model.predict([x, t, cond])
        
        
        # Use our model (noise predictor) to predict the mean of denoised
        model_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)

        #all t's are the same
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = tf.gather(self.posterior_variance, t)
            noise = tf.random.normal(x.shape)
            # Algorithm 2 line 4:
            out = model_mean + tf.math.sqrt(posterior_variance_t) * noise 
            return out



    def Sample(self, cond, num_steps = 200, 
                  gen_batch_size = 256, eps = 1e-3):
        """Generate samples from diffusion model.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """
        import time
        gen_size = cond.shape[0]
        data_shape = np.concatenate(([gen_size],self._data_shape))
        noise = tf.random.normal((data_shape))


        x = noise

        time_steps = list(range(self.nsteps))
        time_steps.reverse()

        start = time.time()
        for time_step in time_steps:      

            batch_time_step = tf.ones((gen_size,1), dtype = tf.dtypes.int32) * time_step
            x = self.p_sample(x, cond, batch_time_step, batch_size = gen_batch_size)


        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start))

        return x




    def PCSampler(self,
                  cond,
                  # num_steps=900, 
                  # snr=0.165,
                  #num_steps=2000,
                  num_steps=200, 
                  #snr=0.23,
                  snr=0.3,
                  ncorrections=1,
                  gen_batch_size = 256,
                  eps=1e-3):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        import time
        batch_size = cond.shape[0]
        t = tf.ones((batch_size,1))
        data_shape = np.concatenate(([batch_size],self._data_shape))
        const_shape = np.concatenate(([batch_size],self.shape[1:]))
        cond = tf.convert_to_tensor(cond, dtype=tf.float32)
        cond = tf.reshape(cond,(-1,self._num_cond))
        init_x = self.prior_sde(data_shape)
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]        
        x = init_x

        if self.sde_type == 'VESDE':
             discrete_sigmas = np.exp(np.linspace(np.log(self.sigma_0), np.log(self.sigma_1), num_steps))
        else:
            discrete_betas = np.linspace(self.beta_0 / num_steps, self.beta_1 / num_steps, num_steps)
            alphas = 1. - discrete_betas

        start = time.time()
        for istep,time_step in enumerate(time_steps):      
            batch_time_step = tf.ones((batch_size,1)) * time_step
            time_idx = num_steps - istep -1
            z = tf.random.normal(x.shape)
            score = self.model.predict([x, batch_time_step,cond], batch_size = gen_batch_size)

            if self.sde_type == 'VESDE':
                alpha = tf.ones(const_shape)
            else:
                alpha = tf.ones(const_shape) *alphas[time_idx]



            for _ in range(ncorrections):
                # Corrector step (Langevin MCMC)
                grad = score
                noise = tf.random.normal(x.shape)
                
                grad_norm = tf.reduce_mean(tf.norm(tf.reshape(grad,(grad.shape[0],-1)),axis=-1,keepdims =True),-1)
                grad_norm = tf.reduce_mean(grad_norm)
                
                noise_norm = tf.reduce_mean(tf.norm(tf.reshape(noise,(noise.shape[0],-1)),axis=-1,keepdims =True),-1)
                noise_norm = tf.reduce_mean(noise_norm)
                
                langevin_step_size = alpha*2 * (snr * noise_norm / grad_norm)**2
                langevin_step_size = tf.reshape(langevin_step_size,self.shape)
                x_mean = x + langevin_step_size * grad
                x =  x_mean + tf.math.sqrt(2 * langevin_step_size) * noise

            
            # Predictor step (Euler-Maruyama)
            
            drift,diffusion = self.sde(x,batch_time_step)
            drift = drift - (diffusion**2) * score     
            x_mean = x - drift * step_size            
            x = x_mean + tf.math.sqrt(diffusion**2*step_size) * z

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))
        # The last step does not include any noise
        return x_mean
