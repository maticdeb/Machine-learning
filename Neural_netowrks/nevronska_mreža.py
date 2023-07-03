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

X = np.load(directory + f"/PodatkiK/intenziteta{int(lbd)}noise{int(0)}.npy")
Y = np.load(directory + f"/PodatkiK/C_values.npy") / (300 * 1e-10)
print(X.shape)


X0 = np.load(directory + f"/PodatkiK/intenziteta{int(lbd)}noise{int(100)}.npy")
Y0 = np.load(directory + f"/PodatkiK/C_values.npy") / (300 * 1e-10)



test_X = X0[90000:]
test_Y = Y0[90000:]
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

model = create_model()

history = model.fit(X,Y, epochs= 400, batch_size=100, validation_split=0.1, shuffle=True ,verbose=2)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


napoved = model.predict(test_X)


print(napoved[0][0])

plt.plot(test_Y,napoved , 'b.', markersize=1)
plt.xlabel(r"$C_{true}[3*10^{10}]$")
plt.ylabel(r"$C_{pred}[3*10^{10}]$")
plt.show()

napacne = []

for i in range(len(test_Y)):
    if abs(test_Y[i]-napoved[i]) > 0.2:
        napacne.append(i)

print(len(napacne))

for j in range(len(napacne)):
    a = test_Y[napacne[j]] * 3 * 10**(-8)
    b = napoved[napacne[j]][0]* 3 * 10**(-8)
    plt.plot(test_X[napacne[j]])
    plt.text(200,0.9,r"$C_{true} = $" + f"{round(a,10)}", fontsize=10)
    plt.text(200,0.8,r"$C_{pred} = $" + f"{round(b,10)}", fontsize=10)
    plt.show()
"""
kf = KFold(n_splits=5, shuffle=True)
fold = 1
ocene = []
for train, test in kf.split(X[:10000], Y[:10000]):
    model_CV = create_model()
    history = model_CV.fit(X[train], Y[train], epochs= 100, batch_size=100, verbose=0)

    score = model_CV.evaluate(X[test], Y[test], verbose=0)
    print(f"MSE folda {fold} = {score}")

    ocene.append(score)
    fold = fold + 1

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