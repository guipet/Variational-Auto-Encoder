import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Add
from tensorflow.keras import backend as K

class Sampling(Layer):
    '''
    Sampling Normal Law (z_mean, z_sigma)
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class Encoder(Layer):
    '''
    VAE Encoder Layer. Generates the outputs: 
    - the mean z_mu (moyenne)
    - the log of variance z_log_var 
    - Simulations of a normal law of parameters z_mu, z_sigma (transforamtion de z_log_var)
    '''
    
    def __init__(self, latent_dim, intermediate_dim = [256], name = 'encoder', **kwargs):
        '''
        Inputs:
        - latent_dim : int. Dimension of the latent space
        - intermediate_dim : list. Dimension of the intermediate layers
        - name : string. Name of the layer
        '''
        super(Encoder, self).__init__(name = name, **kwargs)
        
        #intermediate layers
        self.seq = Sequential()
        for i in intermediate_dim:
            self.seq.add( Dense(i, activation = 'relu') )

        #output
        self.mean = Dense(latent_dim)
        self.log_var = Dense(latent_dim)
        
        #sampling
        self.sampling = Sampling()
        
    def call(self, inputs):
        x = self.seq(inputs)
        z_mu = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mu, z_log_var))
        
        return z_mu, z_log_var, z
        

class Decoder(Layer):
    '''
    Dencode the z variables of the latent space to reconstruct the image
    '''
    
    def __init__(self, original_dim, intermediate_dim = [256], name = 'decoder', **kwargs):
        '''
        Inputs : 
        - original_dim : int. Original size of the inputs
        - intermediate_dim : list. Dimension of the intermediate layers
        - name : string. Name of the layer
        '''
        
        super(Decoder, self).__init__(name = name, **kwargs)
        
        #couches intermédiaires
        self.seq = Sequential()
        for i in intermediate_dim:
            self.seq.add( Dense(i, activation = 'relu') )
        
        #couche finale
        self.final = Dense(original_dim, activation = 'sigmoid')
        
    def call(self, inputs):
        
        x = self.seq(inputs)
        return self.final(x)




