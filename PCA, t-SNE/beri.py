import numpy as np
import matplotlib.pyplot as plt

def spektri():
    spectra_array=[]

    for spectrum in range(1,10000):
        flux = np.loadtxt('spektri/%s.dat' % spectrum, comments = '#')
        spectra_array.append(flux)

    spectra_array=np.array(spectra_array)
    return spectra_array



"""
plt.plot(wav, a[80], 'k-')
plt.xlabel('Wavelength')
plt.ylabel('Normalized flux')
plt.show()
"""