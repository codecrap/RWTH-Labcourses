#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from calibration import ChtoE
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sys
sys.path.append("./../../")
import PraktLib as pl

matplotlib.style.use("../labreport.mplstyle")

# @TODO: entire error calculation

# get data
data = np.loadtxt('photo_peaks.NORM')
E = np.array(data[0])
mean = pl.uarray_tag(data[1], data[2], 'stat')
sig = pl.uarray_tag(data[3], data[4], 'stat')
#sig = ChtoE(sig)
#print(sig)

# calc FWHM
FWHM = 2 * np.sqrt(2 * np.log(2)) * sig
# dFWHM = 2 * np.sqrt(2 * np.log(2)) * dsig

# get channel values at half maximum
right = mean + FWHM/2
left = mean - FWHM/2

# convert to energy values
right = ChtoE(right)
left = ChtoE(left)

# calc FWHM in energy units
FWHM = right-left

# resolution-energy plot
name = 'resolution'
x = E
y = FWHM/E
xval = unp.nominal_values(E)
yval = unp.nominal_values(FWHM/E)
xerr = unp.std_devs(x)
yerr = unp.std_devs(y)

fig, ax = plt.subplots()
ax.plot(xval, yval, '.')
ax.errorbar(xval,yval,xerr=xerr,yerr=yerr,fmt='.',color='b')
ax.set_title(name)
ax.set_xlabel('$E$ [keV]')
ax.set_ylabel(r'$\frac{\Delta E}{E}$')
fig.tight_layout()
#fig.show()
fig.savefig("Figures/"+name+".pdf")


# get a and b
name = 'resolution constants'

def poly(en, a, b):
    return (en*a)**2 + en*b**2

x = E
xval = unp.nominal_values(x)
xerr = unp.std_devs(x)
y = FWHM**2
yval = unp.nominal_values(y)
yerr = unp.std_devs(y)
ystat, ysys = pl.split_error(y)

opt, cov = curve_fit(poly, E, yval, sigma=ystat)
a = ufloat(opt[0], cov[0][0], 'stat')
b = ufloat(opt[1], cov[1][1], 'stat')
fit = poly(, opt[0], opt[1])

fig, ax = plt.subplots()
ax.plot(xval, yval, '.')
ax.errorbar(xval,yval,xerr=xerr,yerr=yerr,fmt='.',color='b')
ax.plot(xval, fit, 'r.')
ax.set_title(name)
ax.set_ylabel(r'$\Delta E^{2}$ [keV]')
ax.set_xlabel(r'$E^{2} \cdot a^{2}+E \cdot b^{2}$')
fig.tight_layout()
fig.show()
fig.savefig("Figures/"+name+".pdf")
