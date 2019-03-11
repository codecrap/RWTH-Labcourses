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
import scipy.optimize as spopt
import uncertainties.unumpy as unp
from uncertainties import ufloat, UFloat
from scipy.constants import h, c, elementary_charge

import sys
sys.path.append("./../../")
import PraktLib as pl

from importlib import reload
pl = reload(pl)

import matplotlib
matplotlib.style.use("../labreport.mplstyle")
plt.ioff()

# will need this def
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
#vMeanErr = np.full(len(vMean), 1/np.sqrt(12))
vMeanErr = np.array(vMeanErr)

# linear fit
X = vMean
X_err = vMeanErr
Y = vEnergy
Y_err = vEnergyErr

def f(B, x):
	return B[0]*x + B[1]
	#return -(B[2]*x - B[0])**2 + B[2]
model  = odr.Model(f)
data   = odr.RealData(X, Y, sx=X_err, sy=Y_err)
odrObject = odr.ODR(data, model, beta0=[1, 1, 1])
output = odrObject.run()
ndof = len(X)-2
chiq = output.res_var
#corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
fitparam = [output.beta[0], output.beta[1]]#, output.beta[2]]
fitparam_err = [output.sd_beta[0], output.sd_beta[1]]#, output.sd_beta[2]]
	
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
#m = fitparam[2]
#m_err = fitparam_err[2]

plt.close('all')

# calibration function
a = ufloat(a, a_err, 'sys')
b = ufloat(b, b_err, 'sys')
#m = ufloat(m, m_err, 'sys')

def chToE(ch):
	if isinstance(ch, UFloat):
		E = a * ch + b
		return E
	else:
		E = []
		for i in range(len(ch)):
			 E += [a * ch[i] + b]
#	if isinstance(ch, UFloat):
#		E = -(m * ch - a)**2 + b
#		return E
#	else:
#		E = []
#		for i in range(len(ch)):
#			 E += [-(m * ch[i] - a)**2 + b]
		return np.array(E)


### ANALYSE FE ###

# get data
vNoise = np.genfromtxt(DATAPATH+"am_spek_leer"+FILE_POSTFIX, skip_header=14, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')
#vNoise = np.genfromtxt(DATAPATH+"am_spek_papier"+FILE_POSTFIX, skip_header=12, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')
vData = np.genfromtxt(DATAPATH+'am_spek_fe'+FILE_POSTFIX, skip_header=14, skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')
vCh = np.arange(len(vData))

# set peak bound
peakBound = [1530, 1590]#[560, 585]#

## plot noise
#fig, ax = plt.subplots()
#ax.plot(vCh, vNoise, 'b.')
#ax.set_xlabel('MCA channel')
#ax.set_ylabel('event counts')
#ax.set_title('background measurement')
#fig.savefig("Figures/am_noise.pdf")
#
## raw plot
#fig, ax = plt.subplots()
#ax.plot(vCh, vData, 'b.')
#ax.set_xlabel('MCA channel')
#ax.set_ylabel('event counts')
#ax.set_title("raw data for steel")
#fig.savefig("Figures/am_fe_raw.pdf")
#
## clean plot
vData = vData - vNoise
##_,vData,_ = np.split(vData, [1000,-1])
##_,vEnergy,_ = np.split(vEnergy, [1000,-1])
#fig, ax = plt.subplots()
#ax.plot(vCh, vData, 'b.')
#ax.set_xlabel('energy [keV]')
#ax.set_ylabel('event counts')
#ax.set_title('clean data for steel')
#fig.savefig("Figures/am_fe_clean.pdf")

# cut out peak
_,vPeakData,_ = np.split(vData, peakBound)
_,vPeakCh,_ = np.split(vCh, peakBound)
#print(len(vPeakData))
#print(len(vPeakE))

# fit gauss curve
def g(x, beta):
	y = []
	for i in range(len(x)):
		y += [beta[2] * np.exp(-(x-beta[0])**2/(2.*beta[1]**2))]
	return np.array(y)

X = vPeakCh
X_err = np.full(len(vPeakCh), 1e-10)
Y = vPeakData
Y_err = np.full(len(vPeakData), 1e-10)

#model  = odr.Model(g)
#data   = odr.RealData(X, Y, sx=X_err, sy=Y_err)
#odrObject = odr.ODR(data, model, beta0=[np.mean(peakBound), 1, 1])
#output = odrObject.run()
#ndof = len(X)-3
#chiq = output.res_var*ndof
#corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
#opt = [output.beta[0], output.beta[1], output.beta[2]]
#cov = [[output.sd_beta[0]**2,0,0], [0,output.sd_beta[1]**2,0], [0,0,output.sd_beta[2]**2]]

#p0 = [np.mean(peakBound),1,1]	# start values
#chifunc = lambda p,x,xerr,y,yerr: (y-g(x,p))/np.sqrt(yerr**2+g(x,p)*(-2*(x-p[0])/(2*p[1]**2))*xerr**2)	# p[0] = d/dx line()
#fitparam,cov,_,_,_ = spopt.leastsq(chifunc,p0,args=(X,X_err,Y,Y_err),full_output=True)
## print(fitparam,cov)
#chiq = np.sum(chifunc(fitparam,X,X_err,Y,Y_err)**2) / (len(Y)-len(fitparam))
#fitparam_err = np.sqrt(np.diag(cov)*chiq)									# leastsq returns the 'fractional covariance matrix'
## print(chiq,fitparam_err)
#opt = [fitparam[0], fitparam[1], fitparam[2]]
#cov = [[fitparam_err[0]**2,0,0], [0,fitparam_err[1]**2,0], [0,0,fitparam_err[2]**2]]

opt, cov = curve_fit(pl.gauss, X, Y, p0=[np.mean(peakBound),1,1])

mean = ufloat(opt[0], np.sqrt(cov[0][0]))
sigma = ufloat(abs(opt[1]), np.sqrt(cov[1][1]))
norm = opt[2]

# plot with fit
fig, ax = plt.subplots()
ax.plot(vCh, pl.gauss(vCh, opt[0], opt[1], opt[2]), 'r-',
	 label=r"$\mu = {:.1ufL}, \sigma = {:.1ufL}$".format(ufloat(opt[0],np.sqrt(cov[0][0])), ufloat(abs(opt[1]), np.sqrt(cov[1][1])) ) )
ax.plot(vCh, vData, 'b.', label='steel data with gauss peak')
ax.legend(loc='upper right')
ax.set_xlabel('MCA channel')
ax.set_ylabel('event counts')
ax.set_title("steel with gauss fit")
fig.savefig("Figures/am_fe_gauss.pdf")

plt.close('all')

# convert to energy
E = chToE(mean)
E_stat, E_sys = pl.split_error(E)

# calc wavelength
#wl = (h*c / (E*1000*elementary_charge)) /1e-9

# print result for energy of x-ray
print('energy of x-ray for steel: (%.3f \pm %.3f \pm %.3f)keV'%(E.n, E_stat, E_sys))
#print('wavelength of x-ray for steel: ({})nm'.format(wl))


### ANALYSE ENERGY RESOLUTION ###

vMean = []
vSigma = []

# set peak bounds
#vPeakBounds = vPeakBounds

for i, source in enumerate(vSOURCES):
	
	# get data
	vData = np.genfromtxt(FILE_PREFIX+source+FILE_POSTFIX, skip_header=vHEADER[i], skip_footer=37, encoding='latin-1', dtype=int, delimiter='\n')
	vCh = np.arange(len(vData))
	
	# cut out peak
	_,vPeakData,_ = np.split(vData, vPeakBounds[i])
	_,vPeakCh,_ = np.split(vCh, vPeakBounds[i])
	
	# fit gauss curve
	opt, cov = curve_fit(pl.gauss, vPeakCh, vPeakData, p0=[np.mean(vPeakBounds[i]),1,1])
	vMean += [ufloat(opt[0], np.sqrt(cov[0][0]), 'stat')]
	vSigma += [ufloat(abs(opt[1]), np.sqrt(cov[1][1]), 'stat')]
	norm = opt[2]
	
vMean = np.array(vMean)
vSigma = np.array(vSigma)

# calc FWHM
vFWHM = 2 * np.sqrt(2 * np.log(2)) * vSigma
vLeft = vMean - vFWHM/2
vRight = vMean + vFWHM/2
vFWHM = chToE(vRight) - chToE(vLeft)

vMean = chToE(vMean)

vMean_stat, vMean_sys = pl.split_error(vMean)
vFWHM_stat, vFWHM_sys = pl.split_error(vFWHM)

# print results
print('resolution results:')
for i in range(len(vMean)):
	print('mean'+str(i)+'= (%.3f \pm %.3f \pm %.3f)keV'%(vMean[i].n, vMean_stat[i], vMean_sys[i]))
	print('FWHM'+str(i)+'= %.3f \pm %.3f \pm %.3f'%(vFWHM[i].n, vFWHM_stat[i], vFWHM_sys[i]))

# plot resolution
vRes = vFWHM/vMean
X = unp.nominal_values(vMean)
X_err = unp.std_devs(vMean)
Y = unp.nominal_values(vRes)
Y_err =	unp.std_devs(vRes)

fig, ax = plt.subplots()
ax.plot(X, Y, '.')
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, fmt='.', color='b')
ax.set_title('resolution')
ax.set_xlabel('energy [keV]')
ax.set_ylabel(r'$\frac{\Delta E}{E}$')
fig.tight_layout()
fig.savefig("Figures/am_resolution.pdf")	
