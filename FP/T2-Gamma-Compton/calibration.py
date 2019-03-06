#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat, UFloat
import peakutils as pu
import operator
from functools import reduce

import sys
sys.path.append("./../../")
import PraktLib as pl

from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

import matplotlib
matplotlib.style.use("../labreport.mplstyle")
plt.ioff()

# Element order: Cs, Na, Co, Eu
vSOURCES = ['Cs', 'Na', 'Co', 'Eu']
DATAPATH = "./Data/"
FILE_POSTFIX = "_calibration.TKA"
vCOLORS = ['r','g','b','k','m','y']

# peak bounds
Cs = [[420,475]]
Na = [[820,865]]
Co = [[750,800], [850,910]]
Eu = [[85,105], [165,190], [225,255], [505,540], [620,665], [890,970]] #[690,780]
# vPeakBounds = reduce(operator.concat,[Cs, Na, Co, Eu])
vPeakBounds = [Cs,Na,Co,Eu]

# expected values
theory = np.array([661.66,
                   1274.5,
                   1173.2, 1332.5,
                   121.78, 244.70, 344.28, 778.90, 964.08, 1408.0]) #1112.1
    
# sort peaks
sort = np.array([3,
                 7,
                 6, 8,
                 0, 1, 2, 4, 5, 9])
    
# get noise
vNoise =  np.genfromtxt(DATAPATH+"Noise"+FILE_POSTFIX, dtype=int, delimiter='\n', skip_header=2)


# set channel array
vCh = np.array(range(len(vNoise)))

# 
# mean = [[0],[0],[0,0],[0,0,0,0,0,0,0]]
vMean = []
# dmean = []
vSigma = []
# dsig = []
n = []
vThres = [0.5,0.1,0.7,0.1]
vMindist = [50,300,80,40]

for i,source in enumerate(vSOURCES):
	
	vData = np.genfromtxt(DATAPATH + source + FILE_POSTFIX, dtype=float, delimiter='\n', skip_header=2)

	# raw plot
	fig, ax = plt.subplots()
	ax.plot(vCh, vData, 'b.')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source+" raw data")
	fig.savefig("Figures/"+source+"_raw")
	
	# plot without noise
	vData -= vNoise
	fig, ax = plt.subplots()
	ax.plot(vCh, vData, 'b.')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source + " without noise")
	fig.savefig("Figures/" + source + "_nonoise")
	
	# plot without baseline estimate
	vBaseline = pu.baseline(vData,deg=8,max_it=200,tol=1e-4)
	vData -= vBaseline
	fig, ax = plt.subplots()
	ax.plot(vCh, vData, 'b.')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source + " without baseline estimate")
	# fig.savefig("Figures/" + source + "_nobaseline")
	
	# # plot with peaks
	# vPeakInds = pu.peak.indexes(vData,thres=vThres[i],min_dist=vMindist[i])
	# fig, ax = plt.subplots()
	# ax.plot(vCh, vData, 'b.')
	# ax.plot(vPeakInds, vData[vPeakInds], 'rx', label="Peakfinder peaks")
	# ax.legend(loc='upper right')
	# ax.set_xlabel('MCA Channel')
	# ax.set_ylabel('Event counts')
	# ax.set_title(source + " without baseline estimate")
	# fig.savefig("Figures/" + source + "_nobaseline")
	
	# plot with fits
	# fig, ax = plt.subplots()
	for j, bound in enumerate(vPeakBounds[i]):
		
		# cut out peaks
		_,vPeakData,_ = np.split(vData, bound)
		_,vPeakCh,_ = np.split(vCh, bound)
		
		# fit gauss curve
		opt, cov = curve_fit(pl.gauss, vPeakCh, vPeakData, p0=[int(np.mean(bound)),1,1])
		vMean += [ufloat(opt[0], np.sqrt(cov[0][0]))]
		# dmean += [np.sqrt(cov[0][0])]
		vSigma += [ufloat(abs(opt[1]), np.sqrt(cov[1][1]))]
		# dsig += [np.sqrt(cov[1][1])]
		n += [opt[2]]
	
		ax.plot(vCh, pl.gauss(vCh, opt[0], opt[1], opt[2]), color=vCOLORS[j],
				label=r"$\mu = {:.1ufL}, \sigma = {:.1ufL}$".format(ufloat(opt[0],np.sqrt(cov[0][0])), ufloat(abs(opt[1]), np.sqrt(cov[1][1])) ) )

	ax.legend(loc='upper right')
	ax.set_xlabel('MCA Channel')
	ax.set_ylabel('Event counts')
	ax.set_title(source + " without baseline estimate")
	fig.savefig("Figures/" + source + "_nobaseline")


plt.close('all')

vMean = np.array(vMean)
# dmean = np.array(dmean)
vSigma = np.array(vSigma)
# dsig = np.array(dsig)
n = [np.array(n)]

# create NORM file
np.savetxt('photo_peaks.NORM', [theory, unp.nominal_values(vMean), unp.std_devs(vMean), unp.nominal_values(vSigma), unp.std_devs(vSigma)])

# sort arrays
mean1 = []
dmean1 = []
theory1 = []
mean2 = []
dmean2 = []
theory2 = []

for i, val in enumerate(sort):
    if val<=4:
        mean1 += [unp.nominal_values(vMean[i])]
        dmean1 += [unp.std_devs(vMean[i])]
        theory1 += [theory[i]]
    else:
        mean2 += [unp.nominal_values(vMean[i])]
        dmean2 += [unp.std_devs(vMean[i])]
        theory2 += [theory[i]]
        
mean1 = np.array(mean1)
dmean1 = np.array(dmean1)
theory1 = np.array(theory1)
mean2 = np.array(mean2)
dmean2 = np.array(dmean2)
theory2 = np.array(theory2)

# linear fits
noerror = np.zeros(len(theory))
fitparam,fitparam_err,chiq = pl.plotFit(unp.nominal_values(vMean), unp.std_devs(vMean), theory, noerror, show=True, title="Calibration fit", xlabel="Channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a = fitparam[0]
da = fitparam_err[0]
b = fitparam[1]
db = fitparam_err[1]

noerror1 = np.zeros(len(theory1))
fitparam1,fitparam_err1,chiq1 = pl.plotFit(mean1, dmean1, theory1, noerror1, show=True, title="Calibration fit 1", xlabel="Channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a1 = fitparam1[0]
da1 = fitparam_err1[0]
b1 = fitparam1[1]
db1 = fitparam_err1[1]

noerror2 = np.zeros(len(theory2))
fitparam2,fitparam_err2,chiq2 = pl.plotFit(mean2, dmean2, theory2, noerror2, show=True, title="Calibration fit 2", xlabel="Channel", ylabel="Energy [keV]", res_ylabel=r"$y - (a \cdot x + b)$")
a2 = fitparam2[0]
da2 = fitparam_err2[0]
b2 = fitparam2[1]
db2 = fitparam_err2[1]

# convertion function
# old
#delim = (min(mean2)-max(mean1))/2
#def ChtoE(ch, dch):
#    if ch<=delim:
#        E = a1 * ch + b1
#        dE = np.sqrt((ch*da1)**2 + (a1*dch)**2 + db1**2)
#    else:
#        E = a2 * ch + b2
#        dE = np.sqrt((ch*da2)**2 + (a2*dch)**2 + db2**2)
#    return [E, dE] # in keV

# new
delim = (min(mean2)-max(mean1))/2
a1 = ufloat(a1, da1, 'sys')
b1 = ufloat(b1, db1, 'sys')
a2 = ufloat(a2, da2, 'sys')
b2 = ufloat(b2, db2, 'sys')
def ChtoE(ch):
	if isinstance(ch, UFloat):
		if ch<=delim: E = a1 * ch + b1
		else: E = a2 * ch + b2
	else:
		for i in range(np.size(ch)): 
			if ch[i]<=delim: E = a1 * ch + b1
			else: E = a2 * ch + b2
	return E # in keV
