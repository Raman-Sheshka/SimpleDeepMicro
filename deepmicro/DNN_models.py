import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Flatten, Reshape, Cropping2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
import numpy as np

class Autoencoder:
    """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    def __init__(self, dims: int, act:str='relu', init:str='glorot_uniform', latent_act:bool=False, output_act:bool=False):
        self.dims = dims
        self.act = act
        self.init = init
        self.latent_act = act if latent_act else None
        self.output_act = 'sigmoid' if output_act else None
        self.encoder, self.autoencoder = self._build_model()
        self.model_repr = self.get_model_string()

    def _build_model(self):
        n_internal_layers = len(self.dims) - 2

        # Input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # Internal layers in encoder
        for i in range(n_internal_layers):
            h = layers.Dense(self.dims[i + 1],
                             activation=self.act,
                             kernel_initializer=self.init,
                             name=f'encoder_{i}')(h)

        # Bottle neck layer
        h = layers.Dense(self.dims[-1],
                         activation=self.latent_act,
                         kernel_initializer=self.init,
                         name=f'encoder_{n_internal_layers}_bottle-neck')(h)

        y = h

        # Internal layers in decoder
        for i in range(n_internal_layers, 0, -1):
            y = layers.Dense(self.dims[i],
                             activation=self.act,
                             kernel_initializer=self.init,
                             name=f'decoder_{i}')(y)

        # Output
        y = layers.Dense(self.dims[0],
                         activation=self.output_act,
                         kernel_initializer=self.init,
                         name='decoder_0')(y)

        autoencoder = models.Model(inputs=x, outputs=y, name='AE')
        encoder = models.Model(inputs=x, outputs=h, name='encoder')

        return encoder, autoencoder

    def get_encoder(self):
        return self.encoder

    def get_autoencoder(self):
        return self.autoencoder

    def get_model_string(self):
        # manipulating an experiment identifier in the output file
        prefix = ""
        if len(self.dims) == 1:
            prefix += "AE_"
        else:
            prefix += "DAE_"
        prefix += "loss_"

        prefix += str(self.latent_act) + "_"
        prefix += str(self.output_act) + "_"
        prefix += str(self.dims).replace(", ", "-") + "_"
        prefix += str(self.act)
        return prefix

class ConvAutoencoder:
    """
        Convolutional auto-encoder model.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder

    """
    def __init__(self, dims:int,
                 act:str='relu',
                 init:str='glorot_uniform',
                 latent_act:bool=False,
                 output_act:bool=False,
                 rf_rate:float=0.1,
                 st_rate:float=0.25):
        self.dims = dims
        self.act = act
        self.init = init
        self.latent_act = act if latent_act else None
        self.output_act = 'sigmoid' if output_act else None
        self.rf_rate = rf_rate
        self.st_rate = st_rate
        self.encoder, self.autoencoder = self._build_model()

    def _build_model(self):
        rf_size = init_rf_size = int(self.dims[0][0] * self.rf_rate)
        stride_size = init_stride_size = int(rf_size * self.st_rate) if int(rf_size * self.st_rate) > 0 else 1
        print("receptive field (kernel) size: %d" % rf_size)
        print("stride size: %d" % stride_size)

        n_internal_layers = len(self.dims) - 1
        if n_internal_layers < 1:
            raise ValueError("The number of internal layers for CAE should be greater than or equal to 1")

        x = Input(shape=self.dims[0], name='input')
        h = x

        rf_size_list = []
        stride_size_list = []

        for i in range(n_internal_layers):
            print("rf_size: %d, st_size: %d" % (rf_size, stride_size))
            h = Conv2D(self.dims[i + 1], (rf_size, rf_size), strides=(stride_size, stride_size), activation=self.act, padding='same', kernel_initializer=self.init, name='encoder_conv_%d' % i)(h)
            rf_size = int(K.int_shape(h)[1] * self.rf_rate)
            stride_size = int(rf_size / 2.) if int(rf_size / 2.) > 0 else 1
            rf_size_list.append(rf_size)
            stride_size_list.append(stride_size)

        reshape_dim = K.int_shape(h)[1:]
        h = Flatten()(h)
        y = h
        y = Reshape(reshape_dim)(y)

        print(rf_size_list)
        print(stride_size_list)

        for i in range(n_internal_layers - 1, 0, -1):
            y = Conv2DTranspose(self.dims[i], (rf_size_list[i-1], rf_size_list[i-1]), strides=(stride_size_list[i-1], stride_size_list[i-1]), activation=self.act, padding='same', kernel_initializer=self.init, name='decoder_conv_%d' % i)(y)

        y = Conv2DTranspose(1, (init_rf_size, init_rf_size), strides=(init_stride_size, init_stride_size), activation=self.output_act, kernel_initializer=self.init, padding='same', name='decoder_1')(y)

        if K.int_shape(x)[1] != K.int_shape(y)[1]:
            cropping_size = K.int_shape(y)[1] - K.int_shape(x)[1]
            y = Cropping2D(cropping=((cropping_size, 0), (cropping_size, 0)), data_format=None)(y)

        autoencoder = models.Model(inputs=x, outputs=y, name='CAE')
        encoder = models.Model(inputs=x, outputs=h, name='encoder')

        return encoder, autoencoder

    def get_encoder(self):
        return self.encoder

    def get_autoencoder(self):
        return self.autoencoder


class VariationalAutoencoder:
    """
        Variational auto-encoder model.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (vae_model, encoder_model), Model of autoencoder and model of encoder
    """
    def __init__(self, dims, act='relu', init='glorot_uniform', output_act=False, recon_loss='mse', beta=1):
        self.dims = dims
        self.act = act
        self.init = init
        self.output_act = 'sigmoid' if output_act else None
        self.recon_loss = recon_loss
        self.beta = beta
        self.encoder, self.decoder, self.vae = self._build_model()

    def _build_model(self):
        n_internal_layers = len(self.dims) - 2

        # Build encoder model
        inputs = Input(shape=(self.dims[0],), name='input')
        h = inputs
        for i in range(n_internal_layers):
            h = Dense(self.dims[i + 1], activation=self.act, kernel_initializer=self.init, name=f'encoder_{i}')(h)
        z_mean = Dense(self.dims[-1], name='z_mean')(h)
        z_sigma = Dense(self.dims[-1], name='z_sigma')(h)
        z = Lambda(self.sampling, output_shape=(self.dims[-1],), name='z')([z_mean, z_sigma])
        encoder = models.Model(inputs, [z_mean, z_sigma, z], name='encoder')

        # Build decoder model
        latent_inputs = Input(shape=(self.dims[-1],), name='z_sampling')
        y = latent_inputs
        for i in range(n_internal_layers, 0, -1):
            y = Dense(self.dims[i], activation=self.act, kernel_initializer=self.init, name=f'decoder_{i}')(y)
        outputs = Dense(self.dims[0], kernel_initializer=self.init, activation=self.output_act)(y)
        decoder = models.Model(latent_inputs, outputs, name='decoder')

        # Instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = models.Model(inputs, outputs, name='vae_mlp')

        # Add loss function
        if self.recon_loss == 'mse':
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.dims[0]
        kl_loss = 1 + K.log(1e-8 + K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma)
        kl_loss = K.sum(kl_loss, axis=-1) * -0.5
        vae_loss = K.mean(reconstruction_loss + (self.beta * kl_loss))
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.metrics_tensors.append(K.mean(reconstruction_loss))
        vae.metrics_names.append("recon_loss")
        vae.metrics_tensors.append(K.mean(self.beta * kl_loss))
        vae.metrics_names.append("kl_loss")

        return encoder, decoder, vae

    @staticmethod
    def sampling(args):
        z_mean, z_sigma = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + z_sigma * epsilon

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_vae(self):
        return self.vae
