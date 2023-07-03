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

lbd = 1000
noise = 0
directory = os.getcwd()

X0 = np.load(directory + f"/PodatkiK13/intenziteta{int(lbd)}noise{int(1000 * noise)}.npy")
Y0 = np.load(directory + f"/PodatkiK13/C_values.npy") / (300 * 1e-10)

test_X = X0[90000:]
test_Y = Y0[90000:]
X0 = X0[:90000]
Y0 = Y0[:90000]

X_multi = []
lambdas = [500, 700, 900, 1000]
for lbd in lambdas:
    X_multi.append(np.load(directory + f"/PodatkiK13/intenziteta{int(lbd)}noise{int(1000 * noise)}.npy"))

print(len(X_multi))    
print(len(X_multi[0]))    

X_multi = np.hstack(X_multi)
print(len(X_multi))    
print(len(X_multi[0]))    
test_X_multi = X_multi[90000:]

X_multi = X_multi[:90000]
def create_model(activation='relu', optimizer='adam'):
    model = Sequential ()
    model.add(Dense (400 , activation =activation, input_shape =(400,)))
    model.add(Dense (200 , activation =activation))
    model.add(Dense (50 , activation =activation))
    model.add(Dense (2, activation ='sigmoid'))
    model.compile(optimizer =optimizer, loss='mean_squared_error', metrics =['mean_absolute_error'])
    return model

def multi_model(activation='relu', optimizer='adam'):
    model = Sequential ()
    model.add(Dense (800 , activation =activation, input_shape =(1600,)))
    model.add(Dense (200 , activation =activation))
    model.add(Dense (100 , activation =activation))
    model.add(Dense (2, activation ='sigmoid'))
    model.compile(optimizer =optimizer, loss='mean_squared_error', metrics =['mean_absolute_error'])
    return model



model = create_model()

history = model.fit(X0,Y0, epochs= 100, batch_size=100, validation_split=0.1, shuffle=True ,verbose=2)

model_multi = multi_model()

zgodovina = model_multi.fit(X_multi,Y0,epochs= 100, batch_size=100, validation_split=0.1, shuffle=True ,verbose=2)


plt.plot(history.history['val_loss'], label=r'ena $ \lambda $')
plt.plot(zgodovina.history['loss'], label='train')
plt.plot(zgodovina.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


napoved = model_multi.predict(test_X_multi)


print(napoved[0][0])

plt.plot(test_Y[:,0],napoved[:,0] , 'b.', markersize=1)
plt.xlabel(r"$C_1^{true}[3*10^{10}]$")
plt.ylabel(r"$C_1^{pred}[3*10^{10}]$")
plt.show()

plt.plot(test_Y[:,1],napoved[:,1] , 'b.', markersize=1)
plt.xlabel(r"$C_3^{true}[3*10^{10}]$")
plt.ylabel(r"$C_3^{pred}[3*10^{10}]$")
plt.show()