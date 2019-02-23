# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

Cs = [[400,490]]
Na = [[810,890]]
Co = [[740,810], [850,910]]
Eu = [[85,105], [160,195], [210,270], [480,560], [610,680], [690,780], [890,980]]

noise = np.genfromtxt("Data/Noise_calibration.TKA")
noise = np.delete(noise, [0,1])

data = np.genfromtxt("Data/Eu_calibration.TKA")
lt = data[0]
rt = data[1]

count = np.delete(data, [0,1])
count = count - noise

plt.plot(count,".")

[before, peak, after] = np.split(count, Eu[3])

chan = np.array(range(len(count)))
[before, seg, after] = np.split(chan, Eu[3])

plt.plot(seg, peak,".")


opt, cov = curve_fit(gauss, seg, peak, p0=[Eu[3][0],1,1])
mean = opt[0]
dmean = np.sqrt(cov[0][0])
sig = opt[1]
dsig = np.sqrt(cov[1][1])
a = opt[2]

print(opt)

plt.plot(gauss(chan,opt[0],opt[1],opt[2]))
