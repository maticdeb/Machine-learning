import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


(x_train, y_train) , _ = mnist.load_data ()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print(x_train[0])

original_dim = np.prod( x_train.shape[1:]) # dimenzija vhodnih podatkov
print(original_dim)
hidden_dim = 64 # skriti sloj z 64 node -i
latent_dim = 2 # 2D latentni prostor
inputs = keras.Input(shape=(original_dim,))
h = keras.layers.Dense (hidden_dim, activation='selu')(inputs)
z_mean = keras.layers.Dense (latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean , z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim) , mean=0.0 , stddev=1.0)
    return z_mean + K.exp( 0.5 * z_log_var ) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean ,z_log_var, z],name ='encoder')


latent_inputs = keras.Input(shape=(latent_dim, ) , name='z_sampling')
x = keras.layers.Dense(hidden_dim , activation ='selu') (latent_inputs)
outputs = keras.layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs ,outputs ,name ='vae')

rec_loss = keras.losses.binary_crossentropy(inputs, outputs )
rec_loss *= original_dim
kl_loss = -0.5*K.sum( 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) ,axis=-1)
vae_loss = K.mean(rec_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile (optimizer ='adam')

batch_size = 32
history = vae.fit(x_train ,x_train ,epochs=200 ,batch_size=batch_size ,validation_data=None)



plt.plot(history.history['loss'], label='train')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()



np.save("wghts.npy", vae.get_weights())


vae.set_weights(np.load ("wghts.npy", allow_pickle = True ) )

x_encoded = encoder.predict(x_train)[2]


N = 15
x = np.linspace(-3, 3,N)
y = np.linspace(-3.5,3.5,N)

plt.scatter(x_encoded[:, 0], x_encoded [:, 1 ], c=y_train, label=y_train)
for j in range(len(y)):
    plt.plot( [x[0], x[-1]], [y[j], y[j]],'r')
for i in range(len(x)):
    plt.plot( [x[i], x[i]], [y[0], y[-1]],'r')
plt.colorbar()
plt.show()


plt.scatter(x_encoded[:, 0], x_encoded [:, 1 ], c=y_train, label=y_train)
plt.colorbar()
plt.show()



images = []

for i in range(len(x)):
    for j in range(len(y)):
        #print(i, j )
        z_sample = np.array([[x[i] , y[j]]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded.reshape (28, 28)
        images.append(digit)
        #plt.matshow(digit)
        #plt.axis('off')
        #plt.savefig(f"{i}_{j}.png", bbox_inches='tight')
        #plt.show()


fig,axes = plt.subplots(N,N,gridspec_kw = {'wspace':0, 'hspace':0})
for i in range(len(x)):
    for j in range(len(y)):
        axes[i,j].imshow(images[N*i + j])
        axes[i,j].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
plt.show()        