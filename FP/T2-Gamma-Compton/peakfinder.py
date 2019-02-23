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

probes = [Cs, Na, Co, Eu]
strings = ['Cs', 'Na', 'Co', 'Eu']

theory = [[661.66],
          [1274.5],
          [1173.2, 1332.5],
          [121.78, 244.70, 344.28, 778.90, 964.08, 1112.1, 1408.0]]

noise = np.genfromtxt('Data/Noise_calibration.TKA')
noise = np.delete(noise, [0,1])

for i, element in enumerate(probes):
    
    data = np.genfromtxt('Data/'+strings[i]+'_calibration.TKA')
    lt = data[0]
    rt = data[1]
    
    count = np.delete(data, [0,1])
    count = count - noise
    
    chan = np.array(range(len(count)))
    
    for bound in element:
        [before, peak, after] = np.split(count, bound)
        [before, seg, after] = np.split(chan, bound)
        
        opt, cov = curve_fit(gauss, seg, peak, p0=[bound[0],1,1])
        mean = opt[0]
        dmean = np.sqrt(cov[0][0])
        sig = opt[1]
        dsig = np.sqrt(cov[1][1])
        a = opt[2]
        
        