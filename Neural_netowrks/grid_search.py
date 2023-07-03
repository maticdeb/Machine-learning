import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


#vpliv števila podatkov na MSE

lbd = 1000
noise = 0
directory = os.getcwd()

X = np.load(directory + f"/PodatkiK/intenziteta{int(lbd)}noise{int(1000 * noise)}.npy")
Y = np.load(directory + f"/PodatkiK/C_values.npy") / (300 * 1e-10)

test_X = X[90000:]
test_Y = Y[90000:]
X = X[:30000]
Y = Y[:30000]

def create_model(activation='sigmoid', optimizer='adam',n1=200,n2=50, activation_out='sigmoid'):
    model = Sequential ()
    model.add(Dense (400 , activation =activation, input_shape =(400,)))
    model.add(Dense (n1 , activation =activation))
    model.add(Dense (n2 , activation =activation))
    model.add(Dense (1, activation =activation_out))
    model.compile(optimizer =optimizer, loss='mean_squared_error', metrics =['mean_absolute_error'])
    return model



modelGS = KerasRegressor(build_fn=create_model, epochs=100, batch_size=100,verbose=0,)

batch_size = [50,100,200]
epochs = [50,100,200]


optimizer = ['SGD', 'adam']
activation =['sigmoid', 'tanh', 'relu']
activation_out =['sigmoid', 'tanh', 'relu']

n1 = [50,100,200,300]
n2 = [50,100,200,300]

param_grid= dict(optimizer=optimizer, activation=activation, activation_out=activation_out)
grid = GridSearchCV(estimator=modelGS, param_grid=param_grid, n_jobs=3, cv=5, scoring='neg_mean_squared_error')
grid.fit(X,Y)

print(f"najboljši rezultat: {grid.best_score_, grid.best_params_}")

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
 
for mean, std, param in zip(means, stds, params):
    print(f"{mean} +- {std} z parametri {param}")