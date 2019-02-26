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

def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

noise = np.genfromtxt('Data/Noise_calibration.TKA')
noise = np.delete(noise, [0,1])

data = np.genfromtxt('Data/Na_calibration.TKA')
lt = data[0]
rt = data[1]

count = np.delete(data, [0,1])
count = count - noise

chan = np.array(range(len(count)))

#plt.plot(chan, count, '.')

bound = [300,390]

[before, peak, after] = np.split(count, bound)
[before, seg, after] = np.split(chan, bound)
        
opt, cov = curve_fit(gauss, seg, peak, p0=[bound[0],1,1])
mean = opt[0]
dmean = np.sqrt(cov[0][0])
sig = opt[1]
dsig = np.sqrt(cov[1][1])
a = opt[2]

#plt.plot(chan, gauss(chan, mean, sig, a))