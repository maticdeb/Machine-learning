import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn . preprocessing import QuantileTransformer
from tensorflow.python.keras.activations import linear


path = 'D:/strojno_ucenje/7Avtoenkoderji/blackbox_jetobs.npy'

x_train = np.load(path)
labels = np.load('D:/strojno_ucenje/7Avtoenkoderji/lhco_10_labels.npy')
x_train = x_train.astype('float32')
f_scaler = QuantileTransformer( output_distribution ='uniform',random_state=100)
x_train_transformed = f_scaler.fit_transform(x_train)
inv_mass =  np.load('D:/strojno_ucenje/7Avtoenkoderji/blackbox_invmass.npy')
print(inv_mass)


"""
ozadje = []
signal = []

for i in range(len(labels)):
    if labels[i] == 0:
        ozadje.append(x_train_transformed[i])
    else:
        signal.append(x_train_transformed[i])

signal = np.array(signal)
ozadje = np.array(ozadje)  
print(len(signal))

m1 = x_train[:,0]
razmerje11 = x_train[:,1]
razmerje12 = x_train[:,2]
md1 = x_train[:,3]
m2 = x_train[:,4]
razmerje21 = x_train[:,5]
razmerje22 = x_train[:,6]
md2 = x_train[:,7]

for i in range(8):

    plt.hist(x_train[:,i],bins=100,histtype=u'step',label='vse')
    plt.hist(signal[:,i],bins=100,histtype=u'step',label='signal')
    plt.hist(ozadje[:,i],bins=100,histtype=u'step',label='ozadje')
    plt.ylabel("N")
    if i == 0:
        plt.xlabel(r"$m_1$")
    elif i == 1:
        plt.xlabel(r'$(\tau_2 / \tau_1)_1$') 
    elif i == 2:
        plt.xlabel(r'$(\tau_3 / \tau_2)_1$')    
    elif i == 3:
        plt.xlabel(r'$m_{d1}$')    
    elif i == 4:
        plt.xlabel(r'$m_2$')    
    elif i == 5:
        plt.xlabel(r'$(\tau_2 / \tau_1)_2$')    
    elif i == 6:
        plt.xlabel(r'$(\tau_3 / \tau_2)_1$')    
    elif i == 7:
        plt.xlabel(r'$m_{d2}$')                               
    plt.legend()
    plt.show()

"""

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

history = vae.fit(x_train_transformed ,x_train_transformed ,epochs=100,batch_size=batch_size ,validation_data=None)

i = 5
np.save(f"blackbox_{i}.npy", vae.get_weights())

x_encoded =  encoder.predict(x_train_transformed)
mean, z_var, napovedan_z = x_encoded 

np.save(f"blackbox_{i}.npy", x_encoded)
a = np.zeros(len(napovedan_z))

"""
for o in range(1,6):
    x_encoded = np.load(f'D:/strojno_ucenje/7Avtoenkoderji/napoved_10_{o}.npy')
    mean, z_var, napovedan_z = x_encoded 

    #plt.scatter(napovedan_z, a, c=labels)
    #plt.colorbar()
    #plt.show()

    ozadje = []
    signal = []

    for i in range(len(labels)):
        if labels[i] == 0:
            ozadje.append(mean[i]**2)
        else:
            signal.append(mean[i]**2)
    ozadje = np.array(ozadje)
    signal = np.array(signal)


    plt.hist(ozadje,60,histtype=u'step',color=f'C{o}',linestyle='dashed', label=f'ozadje {o}')
    plt.hist(np.repeat(signal,10),60,histtype=u'step',color=f'C{o}',label=f'10 x ojačan signal {o}')
plt.xlabel(r"$\bar z ^2$")
plt.ylabel('N')    
plt.legend()
plt.show()


fpr, tpr, thresholds = roc_curve(trues, predictions, pos_label=1)

"""


