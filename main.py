import tensorflow as tf
from tensorflow.keras import backend as K
import argparse
from scipy.misc import imsave
import os

from Model import Network
from Layers import * 
from VAE import VAE 


def parse_args():
	'''
	configuration of the parameters
	'''

    parser = argparse.ArgumentParser(description="VAE Model Tensorflow'")

    parser.add_argument('--results_path', type=str, default='results', help='File path of output images')

    parser.add_argument('--latent_dim', type=int, default=20, help='Dimension of latent space', required = True)

    parser.add_argument('--intermediate_dim_encoder', type=list, default=[500], help='Number of hidden units of the encoder')

    parser.add_argument('--intermediate_dim_decoder', type=list, default=[500], help='Number of hidden units of the decoder')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--plot_latent_space', type=bool, default=False, help='Save the plot of the latent space')

    parser.add_argument('--see_reconstruction', type=bool, default=False, help='Save the plot to test the model')

    parser.add_argument('--plot_2D', type=bool, default=True, help='Plot the latent space in 2 dimensions')


    return check_args(parser.parse_args)

def check_args(args):
	'''
	check the value of the parameters
	'''

	try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass

	try : 
		assert args.latent_dim > 0
	except:
		print('latent_dim must be a int positive')
		return None

	try : 
		assert len(args.intermediate_dim_encoder) > 0
	except:
		print('intermediate_dim_encoder must be a non empty list')
		return None

	try:
		assert len(args.intermediate_dim_decoder) > 0
	except:
		print('intermediate_dim_decoder must be a non empty list')
		return None

	try:
		assert learn_rate > 0
	except:
		print('learn_rate must be positive')
		return None

	try:
		assert num_epochs > 0
	except:
		print('num_epochs must be positive')
		return None
	
	try:
		assert batch_size > 0	
	except:
		print('batch_size muste be positive')
		return None

	try:
		assert type(plot_latent_space) == bool
	except:
		print("plot_latent_space must a boolean")
		return None

	try:
		assert type(see_reconstruction) == bool
	except:
		print("see_reconstruction must be a boolean")
		return None

	if args.plot_2D = True and args.latent_dim != 2 and plot_latent_space = True
		print("If the dimension of the latent space is not 2, the plot will be useless")

	if args.plot_2D = False and args.latent_dim != 3 and plot_latent_space = True
		print("If the dimension of the latent space is not 3, the plot will be useless")
	
	return args



#data 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

#model loss 
def nll(y_true, y_pred):
    '''
    Negative log likelihood (Bernoulli)
    '''

    # keras.losses.binary_crossentropy return the mean 
    # over the last axis. We sum then 
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)




def main(args):
	'''
	implementation of the algorithm
	'''

	#parameters
	directory = args.results_path
	latent_dim = args.latent_dim
	intermediate_dim_encoder = args.intermediate_dim_encoder
	intermediate_dim_decoder = args.intermediate_dim_decoder
	lr = args.learning_rate
	epochs = args.num_epochs
	batch_size = args.batch_size
	plot_latent_space = args.plot_latent_space
	see_reconstruction = args.see_reconstruction
	plot_2D = args.plot_2D



	#instanicate the model 
	vae = VAE()

	#create the network and compile 
	vae_network = vae.model(784, intermediate_dim_encoder = intermediate_dim_encoder, 
                        intermediate_dim_decoder = intermediate_dim_decoder, latent_dim = latent_dim)

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	vae_network.compile(optimizer, loss=nll)


	#fit
	vae_network.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data = (x_test, x_test))

	if plot_latent_space:

		plot_ls = vae.latent_feature(x_test, label = y_test, plot2D = plot_2D)
		imsave(directory + '/latentspace.jpg', plot_ls)

	if see_reconstruction:

		index = tf.random.uniform( shape = [25], minval=0, maxval=10000, dtype=tf.dtypes.int32).numpy()
		data  =  x_test[list(index)].reshape(-1, 28, 28)
		plot_sr  = vae.see_reconstruction(data, (5,5))
		imsave(directory + '/reconstruction.jpg', plot_sr)


if __name__ == '__main__':

	args = parse_args()
	if args is None:
		exist()

	main(args)



