#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:41:29 2019

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("./../../")
import PraktLib as pl

data = np.loadtxt('photo_peaks_2.NORM')
theta = data[0]
mean = data[1]
dmean = data[2]
sig = data[3]
dsig = data[4]

# calc FWHM
FWHM = 2 * np.sqrt(2 * np.log(2)) * sig
dFWHM = 2 * np.sqrt(2 * np.log(2)) * dsig

# set peak bounds
lbound = mean-FWHM/2
rbound = mean+FWHM/2
bounds = [[lbound[i],rbound[i]] for i in range(len(lbound))]

# get counts in peak
m = []
for i in 
[before, peak, after] = np.split(, bounds[i])
temp = np.sum(peak)





