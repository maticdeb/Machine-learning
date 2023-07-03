import numpy as np
import matplotlib.pyplot as plt
from pyjet import DTYPE_PTEPM
from pyjet import cluster

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data = np.array(np.load("D:/strojno_ucenje/6Hadroni/h_bb.npy"))
np.load = np_load_old

print(len(data))

def masa(jet1,jet2):
    p1x = jet1.px
    p1y = jet1.py
    p1z = jet1.pz

    p2x = jet2.px
    p2y = jet2.py
    p2z = jet2.pz
    gibalna = (p1x - p2x) **2 + (p1y - p2y) **2 + (p1z - p2z) **2  
    masa = np.sqrt(gibalna)
    return masa

spomin = []
for i in range(len(data)):
    event = np . array (data[i], dtype = DTYPE_PTEPM )
    sequence = cluster (event, ep=False, R=0.4, p=1)
    incl_jets = sequence.inclusive_jets()

    m = masa(incl_jets[0],incl_jets[1])
    print(m)
    spomin.append(m)

hist ,meje= np.histogram(spomin,100)

plt.hist(spomin,bins=400)
plt.xlim([0,700])
plt.xlabel("masa")
plt.ylabel("N")
plt.show()