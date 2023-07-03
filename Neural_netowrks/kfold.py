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
from statistics import mean

lbd = 1000
noise = 0
directory = os.getcwd()

X = np.load(directory + f"/PodatkiK/intenziteta{int(lbd)}noise{int(1000 * noise)}.npy")
Y = np.load(directory + f"/PodatkiK/C_values.npy") / (300 * 1e-10)
print(X.shape)

test_X = X[90000:]
test_Y = Y[90000:]
print(test_X.shape)
X = X[:90000]
Y = Y[:90000]
print(X.shape)


def create_model(activation='relu', optimizer='adam'):
    model = Sequential ()
    model.add(Dense (400 , activation =activation, input_shape =(400,)))
    model.add(Dense (200 , activation =activation))
    model.add(Dense (50 , activation =activation))
    model.add(Dense (1, activation ='sigmoid'))
    model.compile(optimizer =optimizer, loss='mean_squared_error', metrics =['mean_absolute_error'])
    return model
"""
stevilo_podatkov = [90000, 70000, 50000, 30000, 20000, 10000, 5000, 1000 ]

MSE = []
napaka = []
for i in range(len(stevilo_podatkov)):
    kf = KFold(n_splits=5, shuffle=True)
    fold = 1
    ocene = []

    print(f"stevilo podatkov za učenje: {stevilo_podatkov[i]}")
    for train, test in kf.split(X[:stevilo_podatkov[i]], Y[:stevilo_podatkov[i]]):
        model_CV = create_model()
        history = model_CV.fit(X[train], Y[train], epochs= 150, batch_size=200, verbose=0)

        score = model_CV.evaluate(X[test], Y[test], verbose=0)
        print(f"MSE folda {fold} = {score}")

        ocene.append(score[0])
        fold = fold + 1
    MSE.append(mean(ocene)) 
    napaka.append(np.std(ocene))


plt.plot(stevilo_podatkov,MSE)
plt.errorbar(stevilo_podatkov,MSE,yerr=napaka)
plt.xlabel("stevilo podatkov")
plt.ylabel("MSE")
plt.show()
"""

"""
stevilo_podatkov = [90000, 80000,70000, 60000,50000, 40000, 30000,20000, 10000, 5000, 1000 ]
a=[]
for i in range(len(stevilo_podatkov)):
    X = X[:stevilo_podatkov[i]]
    Y = Y[:stevilo_podatkov[i]]
    model_vse = create_model()
    history = model_vse.fit(X,Y, epochs=200, batch_size=300, validation_split=0.1 ,shuffle=True ,verbose=0)

    ocena = model_vse.evaluate(test_X, test_Y,verbose=0)
    print(f"MSE za {stevilo_podatkov[i]} podatkov: {ocena}")
    a.append(ocena[0])


plt.plot(stevilo_podatkov, a,"ob")
plt.xlabel("stevilo podatkov")
plt.ylabel("MSE")
plt.show()


"""
"""
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

print(f"MSE vsega = {ocena}")
"""

kf = KFold(n_splits=5, shuffle=True)
fold = 1
ocene = []

izgube = []
izgube_val = []
for train, test in kf.split(X[:30000], Y[:30000]):
    model_CV = create_model()
    history = model_CV.fit(X[train], Y[train], epochs= 400, batch_size=100, verbose=0,validation_split=0.1)

    score = model_CV.evaluate(X[test], Y[test], verbose=0)
    print(f"MSE folda {fold} = {score}")

    ocene.append(score[0])
    fold = fold + 1
    izgube.append(history.history['loss'])
    izgube_val.append(history.history['val_loss'])


avg_izgube = []
avg_izgube_val = []

for i in range(len(izgube[0])):
    a = 0
    b = 0
    for j in range(len(izgube)):
        a = a + izgube[j][i]
        b = b + izgube_val[j][i]
    avg_izgube.append(a/len(izgube))  
    avg_izgube_val.append(b/len(izgube_val))  

plt.plot(avg_izgube,label='train')    
plt.plot(avg_izgube_val,label='validation')   
plt.xlabel('epcohs')
plt.ylabel('MSE') 
plt.legend()
plt.show()
