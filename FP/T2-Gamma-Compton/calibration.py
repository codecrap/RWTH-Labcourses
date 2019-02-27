# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
import PraktLib as pl
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# gauss function
def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

# element order: Cs, Na, Co, Eu
strings = ['Cs', 'Na', 'Co', 'Eu']

# peak bounds
Cs = [[400,490]]
Na = [[810,890]]
Co = [[740,810], [850,910]]
Eu = [[85,105], [160,195], [210,270], [480,560], [610,680], [890,980]] #[690,780]
probes = [Cs, Na, Co, Eu]

# expected values
theory = np.array([661.66,
                   1274.5,
                   1173.2, 1332.5,
                   121.78, 244.70, 344.28, 778.90, 964.08, 1408.0]) #1112.1

# get noise
noise = np.genfromtxt('Data/Noise_calibration.TKA')
noise = np.delete(noise, [0,1])

# set channel array
chan = np.array(range(len(noise)))

# 
# mean = [[0],[0],[0,0],[0,0,0,0,0,0,0]]
mean = []
dmean = []
sig = []
dsig = []
n = []

for i, element in enumerate(probes):
    
    # get data
    data = np.genfromtxt('Data/'+strings[i]+'_calibration.TKA')
    count = np.delete(data, [0,1])
    count = count - noise
    
    for j, bound in enumerate(element):
        
        # cut out peaks
        [before, peak, after] = np.split(count, bound)
        [before, seg, after] = np.split(chan, bound)
        
        # fit gauss curve
        opt, cov = curve_fit(gauss, seg, peak, p0=[bound[0],1,1])
        mean += [opt[0]]
        dmean += [np.sqrt(cov[0][0])]
        sig += [opt[1]]
        dsig += [np.sqrt(cov[1][1])]
        n += [opt[2]]
        
mean = np.array(mean)
dmean = np.array(dmean)
sig = np.array(sig)
dsig = np.array(dsig)
n = [np.array(n)]

# create NORM file
file = open('photo_peaks.NORM', 'w')
for i in range(len(mean)):
    file.write('%.2f %.2f %.2f %.2f %.2f \n'%(theory[i],mean[i],dmean[i],sig[i],dsig[i]))
file.close()

# linear fit
noerror = np.zeros(len(theory))
fitparam,fitparam_err,chiq = pl.plotFit(mean, dmean, theory, noerror, title="calibration fit", xlabel="channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a = fitparam[0]
da = fitparam_err[0]
b = fitparam[1]
db = fitparam_err[1]

# convertion function
def ChtoE(ch, dch):
    E = a * ch + b
    dE = np.sqrt((ch*da)**2 + (a*dch)**2 + db**2)
    return [E, dE]