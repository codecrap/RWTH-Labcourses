#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

#efficiency

import numpy as np
from matplotlib import pyplot as plt

# Element order: Cs, Na, Co, Eu
strings = ['Cs', 'Na', 'Co', 'Eu']

# distance between source and detector
r = 0.075 # GUESS. ACTUAL MEASUREMENT MISSING
dr = 0.005

# detector surface used
d = 0.0247 # GUESS BASED ON CLEMENS' MEASUREMENTS
dd = 0.001
F = 2*np.pi * (d/2.)**2
dF = np.pi * d * dd

# activity on day of experiment
A = [18319.5, 859.9, 4590.7, 4592.3] # BETTER IMPLEMENT A READING ROUTINE
dA = [1., 1., 1., 1.] # MISSING

# photon yield (intensity)
I = [[85.0],                                            # Cs
     [99.940],                                          # Na
     [99.85, 99.9826],                                  # Co
     [28.58, 7.580, 26.5, 12.94, 14.60, 13.64, 21.0]]   # Eu

# peaks bounds
bounds = [[[400,490]],
          [[810,890]],
          [[740,810], [850,910]],
          [[85,105], [160,195], [210,270], [480,560], [610,680], [690,780], [890,980]]]

# get noise
noise = np.genfromtxt('Data/Noise_calibration.TKA')
noise = np.delete(noise, [0,1])

## set channel array
#chan = np.array(range(len(noise))) 

for i, valA in enumerate(A):
    # get data
    data = np.genfromtxt('Data/'+strings[i]+'_calibration.TKA')
    count = np.delete(data, [0,1])
    count = count - noise
    
    for j, valI in enumerate(I[i]):
        
        # get counts in peak
        [before, peak, after] = np.split(count, bounds[i][j])
        m = np.sum(peak)
        
        # calc efficiency
        eff = 4*np.pi * r**2 * m /(valA * valI * F)
        deff = 4*np.pi*m/valI * np.sqrt((2*r*dr/(valA*F))**2 + (r**2*dA[i]/(valA**2*F))**2 + (r**2*dF/(valA)*F**2)**2)
        print(str(eff)+' +- '+str(deff))
