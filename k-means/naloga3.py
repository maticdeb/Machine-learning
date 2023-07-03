from time import time
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



#naloga3

def računanje_gibalne(pripadnost, gibalne ,N):
    p = [0] * N
    for i in range(N):
        for j in range(len(pripadnost)):
            if pripadnost[j] == i:
                p[i] += gibalne[j]
    p1, p2  =sorted(p, reverse=True)[:2]

    index1 = p.index(p1)
    index2 = p.index(p2)
    return p1, p2, index1, index2

def masa(p1,p2,ind1, ind2, centri):

    p1x = p1 * np.cos(centri[ind1][1])
    p1y = p1 * np.sin(centri[ind1][1])
    p1z = p1 * np.sinh(centri[ind1][0])

    p2x = p2 * np.cos(centri[ind2][1])
    p2y = p2 * np.sin(centri[ind2][1])
    p2z = p2 * np.sinh(centri[ind2][0])
    gibalna = (p1x - p2x) ** 2 + (p1y - p2y) ** 2 + (p1z - p2z) **2 
    masa = np.sqrt(gibalna)
    return masa


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
podatki = np.array(np.load("D:/strojno_ucenje/6Hadroni/h_bb.npy"))
np.load = np_load_old
#print(podatki[0])
data = np.array(podatki[0])


gibalna = np.array(data[:,0])
eta = np.array(data[:,1])
theta = np.array(data[:,2])
lega = np.vstack((eta,theta)).T
N=10
spomin = []
stetje = []

for h in range(0,430,3):
    stetje.append(h)
    print(h)
    gibalna = np.array(data[:,0])
    eta = np.array(data[:,1])
    theta = np.array(data[:,2])
    lega = np.vstack((eta,theta)).T
    masa_avg = 0
    for i in range(10):
        napoved = KMeans(n_clusters=N).fit(np.array(lega))
        pripadnost = napoved.labels_
        centri = napoved.cluster_centers_
        p1, p2, index1, index2 = računanje_gibalne(pripadnost,data[:,0],N)
        m = masa(p1, p2, index1, index2, centri)
        masa_avg += m
    masa_avg = masa_avg/10  
    spomin.append(masa_avg)
    print(masa_avg)
    for j in range(3):
        index = data[:,0].argmin()
        data = np.delete(data, index,0)



data = np.array(podatki[0])

eta = np.array(data[:,1])
theta = np.array(data[:,2])
lega = np.vstack((eta,theta)).T


N=10
spomin1 = []
for h in range(0,430,3):
    
    eta = np.array(data[:,1])
    theta = np.array(data[:,2])
    lega = np.vstack((eta,theta)).T

    napoved = KMeans(n_clusters=N).fit(np.array(lega))
    pripadnost = napoved.labels_
    centri = napoved.cluster_centers_
    p1, p2, index1, index2 = računanje_gibalne(pripadnost,data[:,0],N)
    m = masa(p1, p2, index1, index2, centri)
    spomin1.append(m)
    for j in range(3):
        index = data[:,0].argmin()
        data = np.delete(data, index,0)



stevilo_odstranjenih = np.arange(0,430,3)


plt.plot(stevilo_odstranjenih,spomin,label="rocni")
plt.plot(stevilo_odstranjenih,spomin1,label="sklearn")
plt.title("N: 10")
plt.xlabel("stevilo odstranjenih delcev")
plt.ylabel("napoved mase")
plt.legend()
plt.show()       

