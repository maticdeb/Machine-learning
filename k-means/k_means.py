import numpy as np
import matplotlib.pyplot as plt


#naloga 1

podatki = np.load("D:/strojno_ucenje/6Hadroni/gauss.npy")
print(podatki.size)
a = podatki[:,0]
print(a)


def k_means(podatki,N):
    a = max(abs(podatki[:,0]))
    b = max(abs(podatki[:,1]))

    centri = []
    for i in range(N):
        centri.append([np.random.uniform(-a,a), np.random.uniform(-b,b)])

    centri = np.array(centri)
    pripadnost = [0] * len(podatki)

    for m in range(10):
        print(m)
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
        for j in range(len(pripadnost)):
            for i in range(len(centri)):
                plt.scatter(centri[i][0],centri[i][1],c=f"r",marker='X',s=150)
                if pripadnost[j] == i:
                    plt.scatter(podatki[j][0], podatki[j][1],c=f"C{i}")
                plt.title(f"korak {m}")
        plt.show()
    return 0

k_means(podatki,4)

