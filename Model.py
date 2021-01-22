from tensorflow.keras.models import Model
import tensorflow as tf
from Layers import * 

class Network(Model):
    '''
    Combines the encoder and the decoder to create a VAE Tensorflow Model
    '''

    def __init__(self, original_dim, intermediate_dim_encoder=[256], intermediate_dim_decoder = [256],
                 latent_dim=32, name="autoencoder", **kwargs):
        '''
        Inputs:
        - original_dim : int. Original size of the inputs
        - intermdiate_dim_encoder / intermediate_dim_decoder : list. Dimension of the intermediate layers of the encoder/decoder
        - latent_dim = int. Dimension of the latent space
        -name : string. Nom du réseau
        '''
        super(Network, self).__init__(name=name, **kwargs)
        
        #encoder
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim_encoder)
        
        #decoder
        self.decoder = Decoder(original_dim = original_dim, intermediate_dim=intermediate_dim_decoder)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean( z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
