import matplotlib.pyplot as plt
from beri import spektri
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

#preberi spektre
x = spektri()
#normalizacija podatkov
x = StandardScaler().fit_transform(x)

#zmanjševanje dimenzij s PCA
pca = PCA(n_components=100)
pca_rezultati = pca.fit_transform(x)

#uproaba TSNE
tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
tsne_pca_rezultati = tsne.fit_transform(pca_rezultati)


tsne1 = tsne_pca_rezultati[:,0]
tsne2 = tsne_pca_rezultati[:,1]

#branje podatkov o vrstah zvezd
tabela = pd.read_csv('vrste.txt', sep= ' ')
stevilo_zvezde = tabela['st']
vrsta_zvezde = tabela['vr']

st = []
oz = []

for i in range(len(stevilo_zvezde)):
    st.append(stevilo_zvezde[i])
    oz.append(vrsta_zvezde[i])

    

#graf
plt.scatter(tsne1, tsne2,s=2)

#vrsta zvezd
for i in range(len(st)):
    plt.text(tsne1[st[i]],tsne2[[st[i]]],f"{oz[i]}")

plt.show()
    
    

