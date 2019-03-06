#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sys
sys.path.append("./../../")
import PraktLib as pl

matplotlib.style.use("../labreport.mplstyle")

# gauss function
def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

# element order: Cs, Na, Co, Eu
strings = ['Cs', 'Na', 'Co', 'Eu']

# peak bounds
Cs = [[420,475]]
Na = [[820,865]]
Co = [[750,800], [850,910]]
Eu = [[85,105], [165,190], [225,255], [505,540], [620,665], [890,970]] #[690,780]
probes = [Cs, Na, Co, Eu]

# expected values
theory = np.array([661.66,
                   1274.5,
                   1173.2, 1332.5,
                   121.78, 244.70, 344.28, 778.90, 964.08, 1408.0]) #1112.1
    
# sort peaks
sort = np.array([3,
                 7,
                 6, 8,
                 0, 1, 2, 4, 5, 9])
    
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
    
    # plot data
    fig, ax = plt.subplots()
    ax.plot(chan, count, '.')
    # plt.show()
    fig.savefig("Figures/"+strings[i]+".pdf",format='pdf',dpi=256)
    
    for j, bound in enumerate(element):
        
        # cut out peaks
        [before, peak, after] = np.split(count, bound)
        [before, seg, after] = np.split(chan, bound)
        
        # fit gauss curve
        opt, cov = curve_fit(gauss, seg, peak, p0=[bound[0],1,1])
        mean += [opt[0]]
        dmean += [np.sqrt(cov[0][0])]
        sig += [abs(opt[1])]
        dsig += [np.sqrt(cov[1][1])]
        n += [opt[2]]
        
mean = np.array(mean)
dmean = np.array(dmean)
sig = np.array(sig)
dsig = np.array(dsig)
n = [np.array(n)]

# create NORM file
np.savetxt('photo_peaks.NORM',[theory,mean,dmean,sig,dsig])

# sort arrays
mean1 = []
dmean1 = []
theory1 = []
mean2 = []
dmean2 = []
theory2 = []

for i, val in enumerate(sort):
    if val<=4:
        mean1 += [mean[i]]
        dmean1 += [dmean[i]]
        theory1 += [theory[i]]
    else:
        mean2 += [mean[i]]
        dmean2 += [dmean[i]]
        theory2 += [theory[i]]
        
mean1 = np.array(mean1)
dmean1 = np.array(dmean1)
theory1 = np.array(theory1)
mean2 = np.array(mean2)
dmean2 = np.array(dmean2)
theory2 = np.array(theory2)

# linear fits
noerror = np.zeros(len(theory))
fitparam,fitparam_err,chiq = pl.plotFit(mean, dmean, theory, noerror, show=False, title="calibration fit", xlabel="channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a = fitparam[0]
da = fitparam_err[0]
b = fitparam[1]
db = fitparam_err[1]

noerror1 = np.zeros(len(theory1))
fitparam1,fitparam_err1,chiq1 = pl.plotFit(mean1, dmean1, theory1, noerror1, show=False, title="calibration fit 1", xlabel="channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a1 = fitparam1[0]
da1 = fitparam_err1[0]
b1 = fitparam1[1]
db1 = fitparam_err1[1]

noerror2 = np.zeros(len(theory2))
fitparam2,fitparam_err2,chiq2 = pl.plotFit(mean2, dmean2, theory2, noerror2, show=False, title="calibration fit 2", xlabel="channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a2 = fitparam2[0]
da2 = fitparam_err2[0]
b2 = fitparam2[1]
db2 = fitparam_err2[1]

# convertion function
# old
#delim = (min(mean2)-max(mean1))/2
#def ChtoE(ch, dch):
#    if ch<=delim:
#        E = a1 * ch + b1
#        dE = np.sqrt((ch*da1)**2 + (a1*dch)**2 + db1**2)
#    else:
#        E = a2 * ch + b2
#        dE = np.sqrt((ch*da2)**2 + (a2*dch)**2 + db2**2)
#    return [E, dE] # in keV

# new
delim = (min(mean2)-max(mean1))/2
a1 = ufloat(a1, da1, 'sys')
b1 = ufloat(b1, db1, 'sys')
a2 = ufloat(a2, da2, 'sys')
b2 = ufloat(b2, db2, 'sys')
def ChtoE(ch):
    for i in range(np.size(ch)): 
        if ch[i]<=delim:
            E = a1 * ch + b1
        else:
            E = a2 * ch + b2
    return E # in keV
