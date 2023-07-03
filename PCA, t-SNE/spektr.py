import numpy as np
import matplotlib.pyplot as plt


flux = np.loadtxt('spektri/1142.dat', comments = '#')

wav = np.loadtxt('spektri/val.dat', comments = '#')

plt.rc('axes', labelsize=22)    # fontsize of the x and y labels

plt.plot(wav, flux, 'k-')
plt.title('HAE',  fontdict = {'fontsize' : 28})

plt.xlabel('Wavelength')
plt.ylabel('Normalized flux')
plt.show()
