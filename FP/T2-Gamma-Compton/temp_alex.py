# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

data = np.genfromtxt("Data/Eu_calibration.TKA")
lt = data[0]
rt = data[1]
data = np.delete(data, 1)
data = np.delete(data, 0)

noise = np.genfromtxt("Data/Noise_calibration.TKA")
noise = np.delete(noise, 1)
noise = np.delete(noise, 0)

count = data - noise

plt.plot(count,".")

chan = np.array(range(len(count)))

def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

opt, cov = curve_fit(gauss, chan, count, p0=[940,5,1])
mean = opt[0]
dmean = np.sqrt(cov[0][0])
sig = opt[1]
dsig = np.sqrt(cov[1][1])
a = opt[2]

print(opt)

plt.plot(gauss(chan,opt[0],opt[1],opt[2]))
