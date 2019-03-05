#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
from matplotlib import pyplot as plt
from calibration import ChtoE
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp


# @TODO: entire error calculation

# get data
data = np.loadtxt('photo_peaks.NORM')
E = unp.uarray(data[0],np.zeros(len(data[0])) )		# @FIXME why taking the theory value and not E+-dE ?
sig = unp.uarray(data[3],data[4])
# dsig = data[4]
sig, dsig = ChtoE(sig, dsig)	# @FIXME not working
# had to kick out negative sigmas that where screwing the calculation
# @TODO: improve calibration, so that maybe we don't have to kick out so much data
sig = np.delete(sig, [0,4,5,6])
E = np.delete(E, [0,4,5,6])

# calc FWHM
FWHM = 2 * np.sqrt(2 * np.log(2)) * sig
# dFWHM = 2 * np.sqrt(2 * np.log(2)) * dsig

# resolution-energy plot
x = 1/np.sqrt(E)
y = FWHM/E

fig, ax = plt.subplots()
ax.plot(x, y, '.')
fig.show()

# get a and b
def poly(e, a, b):
    return (e*a)**2 + e*b**2

y = FWHM**2

opt, cov = curve_fit(poly, E, y)
print(opt)

fig, ax = plt.subplots()
ax.plot(poly(E,opt[0],opt[1]), y, '.')
fig.show()


