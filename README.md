# Variational-Auto-Encoder

The implementation of a Variational Auto Encoder (VAE) based on [this paper](https://arxiv.org/pdf/1312.6114.pdf).

A VAE is a neural network which is supposed to efficiently generate data. The idea is that the inputs are projected in a latent space (encoding part)
and then there are rebuilt from this latent space. This a different from a basic Auto Encoder because a VAE manages the issue from a Bayesian point of view. The encoder and decoder are probabilistic models and they sample data from a law instead of just encoding and decoding the data. 

Here, I have implemented a VAE for the MNIST dataset. The code has been developped in Tensorflow and was created to be the more efficient, general and succinct as possible. There are four files : 
- **Layers.py** : Here are the encoder and the decoder. They are constructed from the *Layer class* of tensorflow.

- **Model.py** : This is the class who construct the architecture. Is is constructed from the *Model class* of tensorflow.

- **VAE.py** : This is a complete class about VAE. It uses **Model.py**, which himself uses **Layers.py**. It constructs a model and, once fitted, we can evaluate the model, generate data and visualize the latent space.

- **main.py** : This is the file to run the model and use all the possibilities offered by **VAE.py**. 

**VAE.ipynb** is a jupyter notebook file to see how to use the VAE class. The only difference is about the plot, where we just plot the figures whereas in the **py** files we save the images.

# Prerequisites
• Tensorflow version 2.5.0 or more recent.  
• Numpy version 1.19.4 or more recent.  
• Plotly version 4.14.1 or more recent.

# Parameters

Here are the parameters :
- `--results_path` : File paths of the output images. *Default* : results.
- `--latent_dim` : Dimension of latent space. *Default* : 20.
- `--intermediate_dim_encoder`: Number of hidden units of the encoder. *Default* : [256].
- `--intermediate_dim_decoder` : Number of hidden units of the decoder. *Default* : [256].
- `--learn_rate`: Learning rate for Adam optimizer. *Default* : 0.001.
- `--num_epochs`: The number of epochs to train the model. *Default*: 20.
- `--batch_size`: Batch size. *Default* : 128.
- `--plot_latent_space`: Save the plot of the latent space. *Default* : False.
- `--see_reconstruction`: Save the plot to test the model. *Default* : False 
- `--plot_2D` : Plot the latent space in 2 dimensions. *Default*: False.

For example, if we want to train our VAE on **30** epochs, with a batch size of **64**, a latent space of dimension **55** and evaluate the model, we run the command:
```
python run_main.py --latent_dim 55 --num_epochs 30 --batch_size 64 --see_reconstruction True
```

# Results
Here are the results of the following commands: 

- `python run_main.py --latent_dim 2 --intermediate_dim_encoder [512, 128, 64] --intermediate_dim_decoder [512, 128, 64] --num_epochs 30 --batch_size 128 --see_reconstruction True --plot_latent_space True --plot2D True`

![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/espace_latent_2D.png)
![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/see_reconstruction_2D.png)

We can notice many things : 
- the network is struggling to distinguish the 4 and the 9. We can understand that when we see the latent space (4 and 9 are superposed, so when the model rebuilt the original data from the latent space, it's hard to know what the data point is)
- The numbers 0, 1, 3, 6 et 8 are esay to discern because they are quite well separeted in the latent space
- The generation of data is quite good for a little model without regularizations (dropout, early stopping...)

There is an example of generation of data for this network:
![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/data_gen_dim2.png)

- `python run_main.py --latent_dim 3 --intermediate_dim_encoder [512, 128, 64] --intermediate_dim_decoder [512, 128, 64] --num_epochs 30 --batch_size 128 --see_reconstruction True --plot_latent_space True --plot2D False`

![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/espace_latent_3D.png)
![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/see_reconstruction_3D.png)

There is an example of generation of the data:
![alt text](https://github.com/guipet/Variational-Auto-Encoder/blob/main/plots/data_gen_dim3.png)

# References

the implementation is based on the work of :  
[1] [https://github.com/hwalsuklee/tensorflow-mnist-VAE](https://github.com/hwalsuklee/tensorflow-mnist-VAE/tree/e7903ac398aa5fc3eec8470177abf6ef78d9cac1)  
[2] [Variational Autoencoders as Generative Models with Keras](https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb)  
[3] [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)  
[4] [A Tutorial on Variational Autoencoders with a Concise Keras Implementation](https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/)  
[5] [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)  
[6] [Tensorflow Example](https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example)  
