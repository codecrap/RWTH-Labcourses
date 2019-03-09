#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:50:00 2019

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
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

# set strings (Element order: ag, ba, cu, mo, rb, tb)
vSOURCES = ['ag', 'ba', 'cu', 'mo', 'rb', 'tb']
vHEADER = [12, 14, 22, 45, 45, 12]
DATAPATH = "./Data/"
FILE_PREFIX = DATAPATH+"am_kal_"
FILE_POSTFIX = ".mca"

# set peak bounds
vPeakBounds = [[1900,2040],[2860,2900],[690,750],[1500,1600],[1160,1220],[3950,3990]]

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






