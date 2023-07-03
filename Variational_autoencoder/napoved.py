from numpy.core.fromnumeric import argsort
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn . preprocessing import QuantileTransformer
from tensorflow.python.keras.activations import linear
from sklearn.metrics import roc_auc_score, roc_curve


path = 'D:/strojno_ucenje/7Avtoenkoderji/blackbox_jetobs.npy'

x_train = np.load(path)
labels = np.load('D:/strojno_ucenje/7Avtoenkoderji/lhco_01_labels.npy')
x_train = x_train.astype('float32')
f_scaler = QuantileTransformer( output_distribution ='uniform',random_state=100)
x_train_transformed = f_scaler.fit_transform(x_train)
inv_mass =  np.load('D:/strojno_ucenje/7Avtoenkoderji/blackbox_invmass.npy')

print(x_train_transformed.shape)
print(x_train_transformed.shape[1:])

original_dim = np.prod( x_train_transformed.shape[1:]) # dimenzija vhodnih podatkov
hidden_dim = 64 # skriti sloj z 64 node -i
latent_dim = 1 # 2D latentni prostor
inputs = keras.Input(shape=(original_dim,))
x = keras.layers.Dense (hidden_dim, activation='selu')(inputs)
x = keras.layers.Dense (hidden_dim, activation='selu')(x)
x = keras.layers.Dense (hidden_dim, activation='selu')(x)
z_mean = keras.layers.Dense (latent_dim, name='z_mean')(x)
z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean , z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim) , mean=0.0 , stddev=1.0)
    return z_mean + K.exp( 0.5 * z_log_var ) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean ,z_log_var, z],name ='encoder')

latent_inputs = keras.Input(shape=(latent_dim, ) , name='z_sampling')
x = keras.layers.Dense(hidden_dim , activation ='selu') (latent_inputs)
x = keras.layers.Dense(hidden_dim , activation ='selu') (x)
x = keras.layers.Dense(hidden_dim , activation ='selu') (x)
outputs = keras.layers.Dense(original_dim)(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs ,outputs ,name ='vae')


rec_loss =keras.losses.mean_squared_error(inputs, outputs)
rec_loss *= 5000
kl_loss = -0.5*K.sum( 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) ,axis=-1)
vae_loss = K.mean(rec_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile (optimizer ='adadelta')

batch_size = 1000

#history = vae.fit(x_train_transformed ,x_train_transformed ,epochs=100,batch_size=batch_size ,validation_data=None)

"""
x_encoded = np.load(f'D:/strojno_ucenje/7Avtoenkoderji/blackbox_1.npy')

mean, z_var, napovedan_z = x_encoded 

ocenjevalec =   z_var


plt.hist(ocenjevalec,100)
plt.show()
"""

x_encoded = np.load(f'D:/strojno_ucenje/7Avtoenkoderji/blackbox_1.npy', allow_pickle=True)

mean, z_var, napovedan_z = x_encoded 
mean2 = mean **2 
index = np.flip(mean2[:,0].argsort())
print(index)
x_train = x_train[index]
a = 800
inv_mass = inv_mass[index]

plt.hist(inv_mass[:a],40,histtype=u'step')
plt.xlabel('inv mass')
plt.ylabel('N')
plt.show()


for i in range(4):

    plt.hist(x_train[:,i][:a],bins=40,histtype=u'step',label='curek 1')
    plt.hist(x_train[:,i + 4][:a],bins=40,histtype=u'step',label='curek 2')

    plt.ylabel("N")
    if i == 0:
        plt.xlabel(r"$m$")
    elif i == 1:
        plt.xlabel(r'$(\tau_2 / \tau_1)$') 
    elif i == 2:
        plt.xlabel(r'$(\tau_3 / \tau_2)$')    
    elif i == 3:
        plt.xlabel(r'$m_{d}$')    
 
    plt.legend()
    plt.show()

