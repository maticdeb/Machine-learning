import numpy as np
import matplotlib.pyplot as plt
from pyjet import DTYPE_PTEPM
from pyjet import cluster
import time

"""
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
podatki = np.array(np.load("D:/strojno_ucenje/6Hadroni/h_bb.npy"))
np.load = np_load_old
#print(podatki[0])
p=1
R=0.4
print(len(podatki[0]))

spomin = []
cas_roka = []
for h in range(0,430,10):
    print(h)
    start = time.time()
    protocurki = np.array(podatki[0])

    for u in range(h):
        index = protocurki[:,0].argmin()
        protocurki = np.delete(protocurki, index,0)
    pravi_curki = []
    while(len(protocurki) > 1):
        matrika = np.zeros((len(protocurki), len(protocurki)))

        for i in range(len(protocurki)):
            for j in range(len(protocurki)):
                if i == j:
                    matrika[i][j] = protocurki[i][0] ** (2*p)
                else:
                    produkt1 = (protocurki[i][1] - protocurki[j][1])**2 +  (protocurki[i][2] - protocurki[j][2])**2 
                    mini = min(protocurki[i][0] ** (2*p), protocurki[j][0] ** (2*p))
                    matrika[i][j] = produkt1 * mini / R ** 2

        ind = np.array(np.where(matrika == matrika.min())).flatten()
        a = ind[:len(ind)//2][0]
        b = ind[len(ind)//2:][0]
        if a < b:
            p_nov = protocurki[a][0] + protocurki[b][0]
            eta_nov = (protocurki[a][0] * protocurki[a][1] + protocurki[b][0] * protocurki[b][1]) / p_nov
            theta_nov = (protocurki[a][0] * protocurki[a][2] + protocurki[b][0] * protocurki[b][2]) / p_nov

            protocurki = np.delete(protocurki, b, 0)
            protocurki = np.delete(protocurki, a, 0)
            protocurki = np.append(protocurki, [[p_nov, eta_nov,theta_nov,1]], axis=0)
            #print("a < b")
        elif b < a:   
            p_nov = protocurki[a][0] + protocurki[b][0]
            eta_nov = (protocurki[a][0] * protocurki[a][1] + protocurki[b][0] * protocurki[b][1]) / p_nov
            theta_nov = (protocurki[a][0] * protocurki[a][2] + protocurki[b][0] * protocurki[b][2]) / p_nov

            protocurki = np.delete(protocurki, a, 0)
            protocurki = np.delete(protocurki, b, 0)
            protocurki = np.append(protocurki, [[p_nov, eta_nov,theta_nov,1]],axis=0)
            #print("a >b")
        else:

            pravi_curki.append( [protocurki[a][0], protocurki[a][1], protocurki[a][2]] )
            protocurki = np.delete(protocurki, a, 0)
            #print("a=b")
    pravi_curki.append( [protocurki[0][0], protocurki[0][1], protocurki[0][2]] )

    pravi_curki = np.array(pravi_curki)
    gibalna = pravi_curki[:,0]
    eta = [pravi_curki[:,1]]
    theta = [pravi_curki[:,2]]
    centri = np.append(eta,theta,axis = 0).T

    p1, p2  =sorted(gibalna, reverse=True)[:2]

    index1 = list(gibalna).index(p1)
    index2 = list(gibalna).index(p2)

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

    m = masa(p1, p2, index1, index2, centri)
    print(m)
    spomin.append(m)
    cas_roka.append(time.time()-start)
"""



def masa1(jet1,jet2):
    p1x = jet1.px
    p1y = jet1.py
    p1z = jet1.pz

    p2x = jet2.px
    p2y = jet2.py
    p2z = jet2.pz
    gibalna = (p1x - p2x) **2 + (p1y - p2y) **2 + (p1z - p2z) **2  
    masa = np.sqrt(gibalna)
    return masa


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data = np.array(np.load("D:/strojno_ucenje/6Hadroni/h_bb.npy"))
np.load = np_load_old


podatki = np.array(data[0])
cas_sklearn = []
st_delcev = len(podatki)
spomin1 = []
delci = []
for i in range(0,430,3):
    delci.append(st_delcev - i)
    event = np.array(podatki, dtype=DTYPE_PTEPM )
    sequence = cluster(event, ep=False, R=0.4, p=1)
    incl_jets = sequence.inclusive_jets()
    index = podatki[:,0].argmin()
    podatki = np.delete(podatki, index,0)

    m = masa1(incl_jets[0],incl_jets[1])
    print(m)
    spomin1.append(m)

st_izlocenih = np.arange(0,430,5)

#cas_sklearn.reverse()

print(delci)



plt.plot(delci,spomin1,label="pyjet algoritem")
plt.legend()
plt.xlabel("stevilo izlocenih delcev")
plt.ylabel("napoved mase")
plt.show()       

"""


for j in range(len(pravi_curki)):
    plt.scatter(pravi_curki[j][1], pravi_curki[j][2],c=f"C{i}",s=20 * pravi_curki[j][0])
plt.show()

"""