from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np


#naloga 2
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
podatki = np.array(np.load("D:/strojno_ucenje/6Hadroni/h_bb.npy"))
np.load = np_load_old
#print(podatki[0])
data = np.array(podatki[0])





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




eta = np.array(data[:,1])
theta = np.array(data[:,2])
lega = np.vstack((eta,theta)).T


N=10
for i in range(10):
    napoved = KMeans(n_clusters=N).fit(np.array(lega))
    pripadnost = napoved.labels_
    centri = napoved.cluster_centers_
    p1, p2, index1, index2 = računanje_gibalne(pripadnost,data[:,0],N)
    m = masa(p1, p2, index1, index2, centri)
    print("masa higgsovega bozona:", m)

