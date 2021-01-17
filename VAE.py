import numpy as np 
import tensorflow as tf
from tensorflow.keras import backend as K
import plotly.express as px
import cv2

from Model import Network
from Layers import * 

class VAE():
    '''
    Create a VAE Tensorflow Model. Once trained, we can:
    - generate data
    - visualize latent space
    - evaluate the quality of the model
    '''
    
    def model(self, original_dim, intermediate_dim_encoder=[64], intermediate_dim_decoder = [64], 
              latent_dim=3, name="autoencoder", **kwargs):
        '''
        Construct the VAE network
        
        Inputs :
        - original_dim : int. Original size of the inputs
        - intermediate_dim_encoder/decoder : list. Dimension of the intermediate layers of the encoder/decoder
        - latent_dim : int.  Dimension of the latent space
        - name : str. Name of the network
        
        Output:
        - VAE Tensorflow Model
        '''

        #keep tracks of useful variables
        self.original_dim = original_dim
        self.latent_dim = latent_dim 

        #generate the network
        self.network = Network(original_dim, intermediate_dim_encoder = intermediate_dim_encoder, 
                              intermediate_dim_decoder = intermediate_dim_decoder ,latent_dim = latent_dim, 
                              name = name, **kwargs)
        
        return self.network  
    
    
    def generate(self, sample = 100):
        '''
        Random sampling of data. 
        Fit the model before!!

        Input : 
        - sample : int. Number of data to generate

        Output:
        - data : tensor. 
        '''

        #retrieve the decoder
        decoder = self.network.decoder
        
        #sample latent variables
        sample = K.random_normal(shape=(sample, self.latent_dim))
        
        #generate observations
        data = decoder(sample)
        
        return data
    

    def latent_feature(self, data, label = None, plot2D = True, x_axis = 'z_1', y_axis = 'z_2', z_axis = 'z_3'):
        '''
        Plot the latent space in 2 (si plot2D = True) or 3 (si plot2D = False) dimensions
        Fit the model before!! 
        Think to rescale the datat to fit with the model
        
        Inputs : 
        - data : array, liste, tensor...
        - label : array, liste, tensor... (useful to distinguish the data on the plot)
        - plot2D : Booléean
        - x/y/z_axis : string. Name of the axes

        Outputs : 
        - figure plotly
        '''

        #retrieve the encoder
        encoder = self.network.encoder
        
        #predict
        _, _, z_test = encoder(data)
        
        #plot
        if plot2D :
            fig = px.scatter(x = z_test[:,0], y = z_test[:,1], color = label, labels={'x':x_axis, 'y':y_axis})
            return fig
            
        else : 
            fig = px.scatter_3d(x = z_test[:,0], y = z_test[:,1], z = z_test[:,2], color = label, 
                          labels={'x':x_axis, 'y':y_axis, 'z': z_axis})
            return fig
     

    def see_reconstruction(self, images, size):
        '''
        Plot side-by-side inputs and predictions to assess network quality
        Fit the model before!! 
        
        Inputs:
        - Images : (nb_image, hauteur, largeur)
        - size : tuple. Nombre de figures par ligne et colonne
        
        Output:
        - figure plotly 
        '''
        
        assert images.shape[0] < size[0]*size[1]
        
        _ , h, w = images.shape
        
        assert h*w == self.original_dim
        
        #table to display images
        img = np.zeros((h*size[0], w*size[1]))
        
        #table to display the reconstruction
        img_pred = np.zeros((h*size[0], w*size[1]))
        
        print('Image Reconstruction...')
        pred = self.network.predict(images.reshape(-1, h*w)).reshape(-1, h, w)
        
        print('Display...')
        for idx, (image, image_pred) in enumerate(zip(images, pred)):
    
            i = int(idx % size[1])
            j = int(idx / size[1])
    
            image_ = cv2.resize(image, dsize = (h_, w_), interpolation=cv2.INTER_CUBIC)
            image_pred_ = cv2.resize(image_pred, dsize = (h_, w_), interpolation=cv2.INTER_CUBIC)
    
            img[j*h_:j*h_+h_,i*w_:i*w_+w_] = image_
            img_pred[j*h_:j*h_+h_,i*w_:i*w_+w_] = image_pred
        
        
        #plot
        fig = px.imshow(np.array([img, img_pred]), facet_col = 0, facet_col_wrap=2, 
                        binary_string = True, title="True data vs Reconstruction data")
        fig.for_each_annotation(lambda a: a.update(text=''))
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        return fig


