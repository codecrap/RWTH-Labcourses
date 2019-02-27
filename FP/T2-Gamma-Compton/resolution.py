#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

# energy resolution

import numpy as np
from matplotlib import pyplot as plt
from calibration import ChtoE

# get data
data = np.loadtxt('photo_peaks.NORM')
E = data[0]
sig = data[3]
dsig = data[4]

# calc FWHM
FWHM = 2 * np.sqrt(2 * np.log(2)) * sig
dFWHM = 2 * np.sqrt(2 * np.log(2)) * dsig

# plot
x = E
y = FWHM/E

plt.plot(x, y)


