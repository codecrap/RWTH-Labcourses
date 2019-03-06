#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from calibration import ChtoE
from scipy.optimize import curve_fit
from scipy import odr
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

def poly(beta, en):
    return (en*beta[0])**2 + en*beta[1]**2

x = E
xerr = np.full(len(E), 1e-15)
y = FWHM**2
yval = unp.nominal_values(y)
yerr = unp.std_devs(y)
ystat, ysys = pl.split_error(y)

model  = odr.Model(poly)
data   = odr.RealData(x, yval, sx=xerr, sy=ystat)
odr    = odr.ODR(data, model, beta0=[1, 1])
output = odr.run()
ndof = len(x)-2
chiq = output.res_var*ndof
corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])

fitparam = [output.beta[0],output.beta[1]]
fitparam_err = [output.sd_beta[0],output.sd_beta[1]]

#opt, cov = curve_fit(poly, E, yval, sigma=ystat)
a = ufloat(fitparam[0], fitparam_err[0], 'stat')
b = ufloat(fitparam[1], fitparam_err[1], 'stat')
print('a = {}'.format(a))
print('b = {}'.format(b))
print('chiq = {}'.format(chiq))
'''
fit = poly([fitparam[0], fitparam[1]], np.arange(min(E),max(E)))

fig, ax = plt.subplots()
ax.plot(np.arange(min(E),max(E)), fit, 'r-')
ax.errorbar(xval,yval,xerr=xerr,yerr=yerr,fmt='.',color='b')
ax.set_title(name)
ax.set_ylabel(r'$(\Delta E)^{2}$ [keV$^2$]')
ax.set_xlabel(r'$E$ [keV]')#^{2} \cdot a^{2}+E \cdot b^{2}$')
fig.tight_layout()
fig.show()
fig.savefig("Figures/"+name+".pdf")
'''

fig,ax = plt.subplots(2,1,figsize=(15,10))
residue = yval-poly(fitparam,x)
ax[0].plot(x,poly(fitparam,x),'r-',
			label="Fit: $a = %.3f \pm %.3f$, \n     $b = %.3f \pm %.3f$"
					% (fitparam[0],fitparam_err[0],fitparam[1],fitparam_err[1]))
ax[0].errorbar(x,yval,xerr=xerr,yerr=yerr,fmt='.',color='b')
ax[0].set_title('title')
ax[0].set_xlabel('xlabel')
ax[0].set_ylabel('ylabel')
ax[0].legend(loc='lower right')
ax[0].grid(True)
ax[1].errorbar(x,residue,yerr=np.sqrt(yerr**2+fitparam[0]*xerr**2),fmt='x',color='b',
			label=r"$\frac{\chi^2}{ndf} = %.3f$" % np.around(chiq,3))
ax[1].axhline(0,color='r')
ax[1].set_title("Residuenverteilung")
ax[1].set_xlabel('xlabel')
ax[1].set_ylabel('resylabel')
ax[1].legend(loc='upper right')
ax[1].grid(True)
fig.tight_layout()
plt.show()
fig.savefig("Figures/resolution_fit.pdf",format='pdf',dpi=256)