#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:44:41 2019

@author: alex
"""

#electron mass
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import c, e
from uncertainties import ufloat
from calibration import ChtoE
import uncertainties.unumpy as unp

import sys
sys.path.append("./../../") # path needed for PraktLib
import PraktLib as pl

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

from importlib import reload # take care of changes in module by manually reloading
pl = reload(pl)
#calibration = reload(calibration)

### with sodium ###

#def gauss(x, x0, sigma, a):
#    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

# read data
noise = np.genfromtxt('Data/Noise_calibration.TKA')
noise = np.delete(noise, [0,1])

data = np.genfromtxt('Data/Na_calibration.TKA')
count = np.delete(data, [0,1])
count = count - noise

chan = np.array(range(len(count)))
# plt.plot(chan, count, '.')

bound = [300,390]

[before, peak, after] = np.split(count, bound)
[before, seg, after] = np.split(chan, bound)
        
opt, cov = curve_fit(pl.gauss, seg, peak, p0=[bound[0],1,1])
mean = opt[0]
dmean = np.sqrt(cov[0][0])
sig = opt[1]
dsig = np.sqrt(cov[1][1])
a = opt[2]
# plt.plot(chan, gauss(chan, mean, sig, a))

# convert to energy value
mean = ufloat(mean, dmean, 'stat')
mean = ChtoE(mean)

# calc electron mass
m = mean*1000*e/c**2
print(m)


### with compton ###

# energy of photon before scattering
E0 = 661.66 

# get angles and energies of scattered photons
theta, mean, dmean, sig, dsig = np.loadtxt('photo_peaks_2.NORM', usecols = (0,1,2,3,4,6,7,8,9,10)) # missing dtheta!
dtheta = np.full(len(theta), 0.)
#print(theta)

mean = pl.uarray_tag(mean, dmean, 'stat')
theta = pl.uarray_tag(pl.degToSr(theta), pl.degToSr(dtheta), 'sys')

# convert to energy values
E = ChtoE(mean)

# linear fit
x = 1 - unp.cos(theta)
y = 1/(E) - 1/(E0)
xval = unp.nominal_values(x)
xerr = unp.std_devs(x)
xstat, xsys = pl.split_error(x)
yval = unp.nominal_values(y)
yerr = unp.std_devs(y)
ystat, ysys = pl.split_error(y)
#print(xval)
#print(xstat)
#print(yval)
#print(ystat)

#plt.plot(xval,yval)

fitparam,fitparam_err,chiq = pl.plotFit(np.array(xval), np.array(xstat), np.array(yval), np.array(ystat),
										title=r"linear fit to find $m_e$",
										xlabel="",
										ylabel="",
										res_ylabel=r"$y - (a \cdot x + b)$",
										capsize=3,fontsize=20,show=True,method='leastsq')

a = ufloat(fitparam[0], fitparam_err[0])
m = e*1000/(a*c**2)
print(m)

