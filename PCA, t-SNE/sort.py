import numpy as np
import matplotlib.pyplot as plt
def spektri():
    wav = np.loadtxt('spektri/val.dat', comments = '#')


    spectra_array=[]
    for spectrum in range(1,10001):
        flux = np.loadtxt('spektri/%s.dat' % spectrum, comments = '#')
        spectra_array.append(flux)

    spectra_array=np.array(spectra_array)
    return spectra_array
"""
plt.plot(wav, spectra_array[5], 'k-')
plt.xlabel('Wavelength')
plt.ylabel('Normalized flux')
plt.show()
"""