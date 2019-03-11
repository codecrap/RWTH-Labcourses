#-*- coding: utf-8 -*-
#
#@xray.py: plot raw data from xray source experiment and perform MCA calibration
#@author: Olexiy Fedorets
#@date: Tue 09.03.2019


import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import peakutils as pu
import scipy.optimize as spo

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl

from matplotlib import pyplot as plt
plt.style.use("../labreport.mplstyle")
# import os
# os.environ["MATPLOTLIBRC"] = "./../"

from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


DATAPATH = "Data/"
FILE_POSTFIX = ".mca"
FILE_PREFIX = "xray_"
vCALIBRATION_SAMPLE = ["cu","mo","ag"]
vEMPTY_MEAS = ["leer_"+x for x in ["leer","folie","papier"] ]
vANALYSIS_SAMPLE = ["10ct","chip","chip2","battery","schnecke","stein","tab","fe"]
vCOMPARISON_SAMPLE = ["pb"+x for x in ["","_PUR","_20kV","_35kV",] ]

#%% MCA CALIBRATION
# order: cu,mo,ag
vSAMPLE_KALPHA_ENERGY = [8.047,17.478,22.162]										# in keV
vPeakBounds = [[1100,1300],[2500,2700],[3200,3500]]
vMean, vSigma = [],[]

fig, ax = plt.subplots(2,2,)
plt.rcParams.update({'font.size': 20})
ax = ax.ravel()

for i,sample in enumerate(vCALIBRATION_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "kal_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[1] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vData -= vNoise
	vCh = np.arange(0,len(vData))
	
	_, vPeakData, _ = np.split(vData, vPeakBounds[i])
	_, vPeakCh, _ = np.split(vCh, vPeakBounds[i])
	
	opt, cov = spo.curve_fit(pl.gauss, vPeakCh, vPeakData, p0=[int(np.mean(vPeakBounds[i])), 1, 1])
	vMean += [ufloat(opt[0], np.sqrt(cov[0][0]))]
	vSigma += [ufloat(abs(opt[1]), np.sqrt(cov[1][1]))]
	
	# define part of gauss peak to show (else logplot is stretched infinitely)
	vPeakfit = pl.gauss(vCh, *opt)
	vPeakMask = 1 < vPeakfit
	
	ax[i].semilogy(vCh[vPeakMask], vPeakfit[vPeakMask], 'r-',
			label=r"$\mu = {:.1ufL}, \sigma = {:.1ufL},$".format(vMean[i],vSigma[i]) + '\n'
				  r"$E_{{K \alpha 1}}$ = {:.3f} keV".format(vSAMPLE_KALPHA_ENERGY[i]) )

	ax[i].legend(loc='upper right')
	ax[i].semilogy(vCh, vData, 'b.', markersize=2)
	ax[i].set_xlabel("MCA Channel")
	ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.upper())

# fig.set_rasterized(True)
fig.delaxes(ax[-1])
fig.savefig("Figures/" + "XRay-calibration-samples")

fitparam,fiterror,chiq = pl.plotFit(unp.nominal_values(vMean),unp.std_devs(vMean),
									vSAMPLE_KALPHA_ENERGY,np.zeros(len(vSAMPLE_KALPHA_ENERGY)),
									x_plotaxis=vCh,
									title="XRay detector calibration",filename="XRay-calibration-fit",
									ylabel="Energy (keV)",xlabel="MCA Channel",
									fitmethod='ODR')
print("Fit: ",fitparam,fiterror,chiq)

def chToE(channel, a=ufloat(fitparam[0],fiterror[0]), b=ufloat(fitparam[1],fiterror[1])):
	return a * channel + b


#%% SAMPLE ANALYSIS
fig, ax = plt.subplots(4, 2, figsize=(30,40))
plt.rcParams.update({'font.size': 20})

ax = ax.ravel()

for i, sample in enumerate(vANALYSIS_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vData -= vNoise
	vCh = np.arange(0, len(vData))
	vE = chToE(vCh)
	ax[i].semilogy(unp.nominal_values(vE), vData, 'b.', markersize=2)
	ax[i].set_xlabel("Energy (keV)")
	ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.upper())

fig.subplots_adjust(hspace=0.001,wspace=0.001)
fig.savefig("Figures/" + "XRay-analysis")


#%% PUR and Voltage EFFECT COMPARISON
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()

for i, sample in enumerate(vCOMPARISON_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vData -= vNoise
	vCh = np.arange(0, len(vData))
	vE = chToE(vCh)
	ax[i].semilogy(unp.nominal_values(vE), vData, 'b.', markersize=2)
	ax[i].set_xlabel("Energy (keV)")
	ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.replace("_"," ").upper())										# escape the underscore to calm down the latex interpreter

fig.savefig("Figures/" + "XRay-comparison")

plt.show()
# plt.close('all')








