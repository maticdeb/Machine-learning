import matplotlib.pyplot as plt
from beri import spektri
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


x = spektri()
print(f" dimenzija x:{len(x[0])}")
print(f"stevilo zvezd x:{len(x)}")

b = []
for i in range(len(x)):
    b.append(x[i][800:])

b_arr = np.array(b)
print(f" dimenzija b:{len(b[0])}")
print(f"stevilo zvezd b:{len(b)}")
"""
y = np.loadtxt('spektri/val.dat', comments = '#')

plt.plot(y, x[80], 'k-')
plt.xlabel('Wavelength')
plt.ylabel('Normalized flux')
plt.show()
"""

b = StandardScaler().fit_transform(b)

pca = PCA(n_components=100)
pca_rezultati = pca.fit_transform(b)


tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
tsne_pca_rezultati = tsne.fit_transform(pca_rezultati)

#print(tsne_pca_rezultati)

tsne1 = tsne_pca_rezultati[:,0]
tsne2 = tsne_pca_rezultati[:,1]


tabela = pd.read_csv('vrste.txt', sep= ' ')
stevilo_zvezde = tabela['st']
vrsta_zvezde = tabela['vr']


st = []
oz = []

for i in range(len(stevilo_zvezde)):
    st.append(stevilo_zvezde[i])
    oz.append(vrsta_zvezde[i])

    


plt.scatter(tsne1, tsne2,s=2)


for i in range(len(st)):
    plt.text(tsne1[st[i]-1],tsne2[st[i]-1],f"{oz[i]}")

plt.show()
"""

principalDf = pd.DataFrame(data = pca_rezultati
             , columns = ['0', '1', '2', '3', '4', '5','6', '7', '8'])

print(principalDf)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1,998):
    ax.scatter(principalDf.values[i][0], principalDf.values[i][1],principalDf.values[i][2], s=10,c="blue")         
plt.show()


for i in range(1,998):
    plt.plot(principalDf.values[i][0], principalDf.values[i][1],'b.')         

plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.show()

for i in range(1,998):
    plt.plot(principalDf.values[i][0], principalDf.values[i][2],'b.')         

plt.xlabel('PCA 0')
plt.ylabel('PCA 2')
plt.show()


for i in range(1,998):
    plt.plot(principalDf.values[i][1], principalDf.values[i][2],'b.')         

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


for i in range(1,998):
    plt.plot(principalDf.values[i][2], principalDf.values[i][3],'b.')         

plt.xlabel('PCA 2')
plt.ylabel('PCA 3')
plt.show()

for i in range(1,998):
    plt.plot(principalDf.values[i][4], principalDf.values[i][5],'b.')         

plt.xlabel('PCA 4')
plt.ylabel('PCA 5')
plt.show()

for i in range(1,998):
    plt.plot(principalDf.values[i][6], principalDf.values[i][7],'b.')         

plt.xlabel('PCA 6')
plt.ylabel('PCA 7')
plt.show()


"""
