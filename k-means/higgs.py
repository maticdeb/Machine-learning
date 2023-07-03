import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def k_means(podatki,N):
    a = max(abs(podatki[:,0]))
    b = max(abs(podatki[:,1]))

    centri = []
    for i in range(N):
        centri.append([np.random.uniform(-a,a), np.random.uniform(-b,b)])

    centri = np.array(centri)
    pripadnost = [0] * len(podatki)

    for m in range(30*N):
        for j in range(len(podatki)):
            max_razdalja = 100000
            for i in range(len(centri)):
                razdalja = np.linalg.norm(podatki[j] - centri[i])
                if razdalja < max_razdalja:
                    pripadnost[j] = i
                    max_razdalja = razdalja
        for i in range(len(centri)):
            števec = 1
            kord1 = centri[i][0]
            kord2 = centri[i][1]
            for j in range(len(pripadnost)):
                if pripadnost[j] == i:
                    števec += 1
                    kord1 += podatki[j][0]
                    kord2 += podatki[j][1]
            kord1 = kord1 / števec
            kord2 = kord2 / števec
            centri[i][0] = kord1        
            centri[i][1] = kord2        
    """
    for j in range(len(pripadnost)):
        for i in range(len(centri)):
            plt.scatter(centri[i][0],centri[i][1],c=f"r",marker='X',s=50)
            if pripadnost[j] == i:
                plt.scatter(podatki[j][0], podatki[j][1],c=f"C{i}")

    plt.show()
    """

    return pripadnost, centri


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
    gibalna = (p1x - p2x) **2 + (p1y - p2y)**2 + (p1z - p2z)**2 
    masa = np.sqrt(gibalna )
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

history = []
for j in range(2,11):
    avg = 0
    print(j)
    for i in range(10):
        napoved = KMeans(n_clusters=j).fit(np.array(lega))
        pripadnost = napoved.labels_
        centri = napoved.cluster_centers_
        p1, p2, index1, index2 = računanje_gibalne(pripadnost,data[:,0],j)
        m = masa(p1, p2, index1, index2, centri)
        print(m)
        avg += m
    avg = avg/10    
    history.append(avg)

spomin = []
for j in range(2,11):
    avg = 0
    print(j)
    for i in range(10):
        pripadnost, centri=k_means(lega,j)
        p1, p2, index1, index2 = računanje_gibalne(pripadnost,data[:,0],j)
        m = masa(p1, p2, index1, index2, centri)
        avg += m
    avg = avg/10
    spomin.append(avg)



plt.plot(np.arange(2,11),spomin,"ko-", label="rocni model")   
plt.plot(np.arange(2,11),history,"bo-",label="sklearn model")   
plt.legend()
plt.xlabel("število gruč")
plt.ylabel("masa")
plt.show() 
