#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:50:00 2019

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import uncertainties.unumpy as unp
from uncertainties import ufloat, UFloat

import sys
sys.path.append("./../../")
import PraktLib as pl

from importlib import reload
pl = reload(pl)

import matplotlib
matplotlib.style.use("../labreport.mplstyle")
plt.ioff()


### CALIBRATION ###

# set strings (Element order: ag, ba, cu, mo, rb, tb)
vSOURCES = ['ag', 'ba', 'cu', 'mo', 'rb', 'tb']
vHEADER = [12, 14, 22, 45, 45, 12]
DATAPATH = "./Data/"
FILE_PREFIX = DATAPATH+"am_kal_"
FILE_POSTFIX = ".mca"

# set peak bounds
vPeakBounds = [[1900,2040],[2860,2900],[690,750],[1500,1600],[1160,1220],[3950,3990]]

# set energy for peaks in keV
vEnergy = np.array([22.10, 32.06, 8.04, 17.44, 13.37, 44.23])
vEnergyErr = np.full(len(vEnergy), 1e-15)

# get noise
vNoise =  np.genfromtxt(FILE_PREFIX+"leer"+FILE_POSTFIX, skip_header=12, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')

# set channel array
vCh = np.array(range(len(vNoise)))

# set lists for fit results
vMean = []
vMeanErr = []
vSigma = []
vSigmaErr = []
vNorm = []

for i, source in enumerate(vSOURCES):
	
	vData = np.genfromtxt(FILE_PREFIX+source+FILE_POSTFIX, skip_header=vHEADER[i], skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')

	# raw plot
	fig, ax = plt.subplots()
	ax.plot(vCh, vData, 'b.')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source+" raw data")
	fig.savefig("Figures/am_"+source+"_raw.pdf")
	
#	# plot without noise
#	vData -= vNoise
#	fig, ax = plt.subplots()
#	ax.plot(vCh, vData, 'b.')
#	ax.set_xlabel('MCA Channel')
#	ax.set_ylabel('Event counts')
#	ax.set_title(source+" without noise")
#	fig.savefig("Figures/"+source+"_nonoise")
	
	# cut out peak
	_,vPeakData,_ = np.split(vData, vPeakBounds[i])
	_,vPeakCh,_ = np.split(vCh, vPeakBounds[i])
	
	# fit gauss curve
	opt, cov = curve_fit(pl.gauss, vPeakCh, vPeakData, p0=[int(np.mean(vPeakBounds[i])),1,1])
	vMean += [opt[0]]
	vMeanErr += [np.sqrt(cov[0][0])]
	vSigma += [abs(opt[1])]
	vSigmaErr += [np.sqrt(cov[1][1])]
	vNorm += [opt[2]]
	
	# plot with fit
	fig, ax = plt.subplots()
	ax.plot(vCh, pl.gauss(vCh, opt[0], opt[1], opt[2]), 'r-',
		 label=r"$\mu = {:.1ufL}, \sigma = {:.1ufL}$".format(ufloat(opt[0],np.sqrt(cov[0][0])), ufloat(abs(opt[1]), np.sqrt(cov[1][1])) ) )
	ax.plot(vCh, vData, 'b.', label='raw data')
	ax.legend(loc='upper right')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source+" with gauss fit")
	fig.savefig("Figures/am_"+source+"_gauss.pdf")

plt.close('all')

vMean = np.array(vMean)
vMeanErr = np.array(vMeanErr)

# linear fit
X = vMean
X_err = vMeanErr
Y = vEnergy
Y_err = vEnergyErr

def f(B, x):
	return B[0]*x + B[1]
model  = odr.Model(f)
data   = odr.RealData(X, Y, sx=X_err, sy=Y_err)
odrObject = odr.ODR(data, model, beta0=[1, 1])
output = odrObject.run()
ndof = len(X)-2
chiq = output.res_var*ndof
corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
fitparam = [output.beta[0],output.beta[1]]
fitparam_err = [output.sd_beta[0],output.sd_beta[1]]
	
fig,ax = plt.subplots(2,1)
residue = Y - f(fitparam, X)
ax[0].plot(X, f(fitparam, X), 'r-',
			label="Fit: $a = %.1e \pm %.1e$, \n     $b = %.1e \pm %.1e$"
					% (fitparam[0],fitparam_err[0],fitparam[1],fitparam_err[1]))
ax[0].errorbar(X, Y, xerr=X_err, yerr=Y_err, fmt='.', color='b')
ax[0].set_title('calibration fit')
ax[0].set_ylabel('Energy [keV]')
ax[0].legend(loc='lower right')
ax[0].grid(True)
ax[1].errorbar(X, residue, yerr=np.sqrt(Y_err**2 + fitparam[0]*X_err**2), fmt='o', color='b',
			label=r"$\frac{\chi^2}{ndf} = %.3f$" % np.around(chiq,3))
ax[1].axhline(0,color='r')
ax[1].set_title("Residuals")
ax[1].set_xlabel('Channel')
ax[1].set_ylabel(r"$y - (a \cdot x + b)$")
ax[1].legend(loc='upper right')
ax[1].grid(True)
fig.tight_layout()
fig.savefig('Figures/'+'am_calibration_fit'+'.pdf')

a = fitparam[0]
a_err = fitparam_err[0]
b = fitparam[1]
b_err = fitparam_err[1]


X = np.delete(vMean, [3, 4, 5])
X_err = np.delete(vMeanErr, [3, 4, 5])
Y = np.delete(vEnergy, [3, 4, 5])
Y_err = np.delete(vEnergyErr, [3, 4, 5])

model  = odr.Model(f)
data   = odr.RealData(X, Y, sx=X_err, sy=Y_err)
odrObject = odr.ODR(data, model, beta0=[1, 1])
output = odrObject.run()
ndof = len(X)-2
chiq = output.res_var*ndof
corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
fitparam = [output.beta[0],output.beta[1]]
fitparam_err = [output.sd_beta[0],output.sd_beta[1]]
	
fig,ax = plt.subplots(2,1)
residue = Y - f(fitparam, X)
ax[0].plot(X, f(fitparam, X), 'r-',
			label="Fit: $a = %.1e \pm %.1e$, \n     $b = %.1e \pm %.1e$"
					% (fitparam[0],fitparam_err[0],fitparam[1],fitparam_err[1]))
ax[0].errorbar(X, Y, xerr=X_err, yerr=Y_err, fmt='.', color='b')
ax[0].set_title('calibration fit')
ax[0].set_ylabel('Energy [keV]')
ax[0].legend(loc='lower right')
ax[0].grid(True)
ax[1].errorbar(X, residue, yerr=np.sqrt(Y_err**2 + fitparam[0]*X_err**2), fmt='o', color='b',
			label=r"$\frac{\chi^2}{ndf} = %.3f$" % np.around(chiq,3))
ax[1].axhline(0,color='r')
ax[1].set_title("Residuals")
ax[1].set_xlabel('Channel')
ax[1].set_ylabel(r"$y - (a \cdot x + b)$")
ax[1].legend(loc='upper right')
ax[1].grid(True)
fig.tight_layout()
fig.savefig('Figures/'+'am_calibration_fit_2'+'.pdf')

a1 = fitparam[0]
a1_err = fitparam_err[0]
b1 = fitparam[1]
b1_err = fitparam_err[1]


X = np.delete(vMean, [0, 1, 2])
X_err = np.delete(vMeanErr, [0, 1, 2])
Y = np.delete(vEnergy, [0, 1, 2])
Y_err = np.delete(vEnergyErr, [0, 1, 2])

model  = odr.Model(f)
data   = odr.RealData(X, Y, sx=X_err, sy=Y_err)
odrObject = odr.ODR(data, model, beta0=[1, 1])
output = odrObject.run()
ndof = len(X)-2
chiq = output.res_var*ndof
corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
fitparam = [output.beta[0],output.beta[1]]
fitparam_err = [output.sd_beta[0],output.sd_beta[1]]
	
fig,ax = plt.subplots(2,1)
residue = Y - f(fitparam, X)
ax[0].plot(X, f(fitparam, X), 'r-',
			label="Fit: $a = %.1e \pm %.1e$, \n     $b = %.1e \pm %.1e$"
					% (fitparam[0],fitparam_err[0],fitparam[1],fitparam_err[1]))
ax[0].errorbar(X, Y, xerr=X_err, yerr=Y_err, fmt='.', color='b')
ax[0].set_title('calibration fit')
ax[0].set_ylabel('Energy [keV]')
ax[0].legend(loc='lower right')
ax[0].grid(True)
ax[1].errorbar(X, residue, yerr=np.sqrt(Y_err**2 + fitparam[0]*X_err**2), fmt='o', color='b',
			label=r"$\frac{\chi^2}{ndf} = %.3f$" % np.around(chiq,3))
ax[1].axhline(0,color='r')
ax[1].set_title("Residuals")
ax[1].set_xlabel('Channel')
ax[1].set_ylabel(r"$y - (a \cdot x + b)$")
ax[1].legend(loc='upper right')
ax[1].grid(True)
fig.tight_layout()
fig.savefig('Figures/'+'am_calibration_fit_1'+'.pdf')

a2 = fitparam[0]
a2_err = fitparam_err[0]
b2 = fitparam[1]
b2_err = fitparam_err[1]

plt.close('all')

# calibration function
a = ufloat(a, a_err, 'sys')
b = ufloat(b, b_err, 'sys')
a1 = ufloat(a1, a1_err, 'sys')
b1 = ufloat(b1, b1_err, 'sys')
a2 = ufloat(a2, a2_err, 'sys')
b2 = ufloat(b2, b2_err, 'sys')
delim = (vMean[3]-vMean[2])/2

def chToE(ch):
	E = a * ch + b
#	if isinstance(ch, UFloat):
#		if ch<=delim: E = a1 * ch + b1
#		else: E = a2 * ch + b2
#	else:
#		for i in range(np.size(ch)): 
#			if ch[i]<=delim: E = a1 * ch + b1
#			else: E = a2 * ch + b2
	return E # in keV


### ANALYSE FE ###

# get data
vNoise = np.genfromtxt(DATAPATH+"am_spek_papier"+FILE_POSTFIX, skip_header=12, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')
vData = np.genfromtxt(DATAPATH+'am_spek_fe'+FILE_POSTFIX, skip_header=14, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')

# set peak bound
peakBound = [1530, 1590]

# plot noise
fig, ax = plt.subplots()
ax.plot(vCh, vNoise, 'b.')
ax.set_xlabel('MCA channel')
ax.set_ylabel('event counts')
ax.set_title('background measurement')
fig.savefig("Figures/am_noise.pdf")

# raw plot
fig, ax = plt.subplots()
ax.plot(vCh, vData, 'b.')
ax.set_xlabel('MCA channel')
ax.set_ylabel('event counts')
ax.set_title("raw data for steel")
fig.savefig("Figures/am_fe_raw.pdf")

# clean plot
vData = vData - vNoise
#_,vData,_ = np.split(vData, [1000,-1])
#_,vCh,_ = np.split(vCh, [1000,-1])
fig, ax = plt.subplots()
ax.plot(vCh, vData, 'b.')
ax.set_xlabel('MCA channel')
ax.set_ylabel('event counts')
ax.set_title('clean data for steel')
fig.savefig("Figures/am_fe_clean.pdf")

# cut out peak
_,vPeakData,_ = np.split(vData, peakBound)
_,vPeakCh,_ = np.split(vCh, peakBound)

# fit gauss curve
opt, cov = curve_fit(pl.gauss, vPeakCh, vPeakData, p0=[int(np.mean(peakBound)),1,1])
mean = opt[0]
meanErr = np.sqrt(cov[0][0])
sigma = abs(opt[1])
sigmaErr = np.sqrt(cov[1][1])
norm = opt[2]

# plot with fit
fig, ax = plt.subplots()
ax.plot(vCh, pl.gauss(vCh, opt[0], opt[1], opt[2]), 'r-',
	 label=r"$\mu = {:.1ufL}, \sigma = {:.1ufL}$".format(ufloat(opt[0],np.sqrt(cov[0][0])), ufloat(abs(opt[1]), np.sqrt(cov[1][1])) ) )
ax.plot(vCh, vData, 'b.', label='raw data')
ax.legend(loc='upper right')
ax.set_xlabel('MCA channel')
ax.set_ylabel('event counts')
ax.set_title("steel with gauss fit")
fig.savefig("Figures/am_fe_gauss.pdf")

plt.close('all')

# convert to energy value
E = chToE(ufloat(mean, meanErr, 'stat'))
sigE = chToE(ufloat(sigma, sigmaErr, 'stat'))

# print result for energy of x-ray
print('energy of x-ray for steel: {}'.format(E))