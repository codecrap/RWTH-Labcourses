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
import operator
from functools import reduce
from textwrap import wrap

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
vCOMPARISON_SAMPLE = ["pb"+x for x in ["_50kV","_PUR","_20kV","_35kV",] ]

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

	ax[i].semilogy(vCh, vData, 'b.', markersize=2)
	ax[i].legend(loc='upper right')
	ax[i].set_xlabel("MCA Channel")
	ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.upper())

# fig.set_rasterized(True)
fig.delaxes(ax[-1])
fig.savefig("Figures/" + "XRay-calibration-samples")
vMean = np.array(vMean)
vSigma = np.array(vSigma)

fitparam,fiterror,chiq = pl.plotFit(unp.nominal_values(vMean),unp.std_devs(vMean),
									vSAMPLE_KALPHA_ENERGY,np.zeros(len(vSAMPLE_KALPHA_ENERGY)),
									x_plotaxis=vCh,
									title="XRay detector calibration",filename="XRay-calibration-fit",
									ylabel="Energy (keV)",xlabel="MCA Channel",
									fitmethod='ODR')
print("Fit: ",fitparam,fiterror,chiq)

def chToE(channel, a=ufloat(fitparam[0],fiterror[0],tag='sys'), b=ufloat(fitparam[1],fiterror[1],tag='sys')):
	return a * channel + b

#%% DETECTOR RESOLUTION
vE = chToE(vMean)
vResolution = pl.stdToFWHM(vSigma)/vE

fig, ax = plt.subplots()
ax.errorbar(unp.nominal_values(vE), unp.nominal_values(vResolution),
			xerr=unp.std_devs(vE), yerr=unp.std_devs(vResolution),
			fmt='o', color='b', ms=20)
ax.set_title('XRay detector resolution')
ax.set_xlabel('Energy (keV)')
ax.set_ylabel(r'$\frac{\Delta E}{E}$')
fig.savefig("Figures/xray_resolution.pdf")



#%% SAMPLE ANALYSIS
fig, ax = plt.subplots(4, 2, figsize=(30,40), sharex='all', sharey='all')
ax = ax.ravel()
plt.rcParams.update({'font.size': 25})

vNEGLECT_TAIL = [30,40,30,20,20,17,40,30]											# energy of bremsstarhlung start in keV
vTHRESHOLD = [120,200,180,200,70,20,180,300]
vMINDIST = [80,110,300,30,200,200,200,70]
vDetectedPeaks = []

for i,sample in enumerate(vANALYSIS_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=12, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=12, skip_footer=71, encoding='latin1')
	vData -= vNoise
	vCh = np.arange(0, len(vData))
	vE = chToE(vCh)
	ax[i].semilogy(unp.nominal_values(vE), vData, 'b.', markersize=2)
	
	# neglect bremsstrahlung part for peakfinder
	vMask = vE<vNEGLECT_TAIL[i]
	vData = vData[vMask]
	vE = vE[vMask]
	
	# neglect front part for 'stein'
	if 'stein' == sample:
		vMask = vE>3
		vData = vData[vMask]
		vE = vE[vMask]
	
	# moving peak search
	vPeakInds = []
	vSliceInds = np.linspace(0,len(vData),2,dtype=int)
	for j,slice in enumerate(vSliceInds[1:]):
		# print(vData[vSliceInds[i]:slice])
		vSlicePeakInds = pu.peak.indexes(vData[vSliceInds[j]:slice],thres=vTHRESHOLD[i],min_dist=vMINDIST[i],thres_abs=True)
		vPeakInds += [vSliceInds[j] + vSlicePeakInds]
	vPeakInds = np.concatenate(vPeakInds)
	
	vDetectedPeaks += [unp.nominal_values(vE[vPeakInds])]
	
	pl.printAsLatexTable(np.array([['${:.2f}$'.format(x) for _, x in enumerate(unp.nominal_values(vE[vPeakInds]))],
								   ['element']*len(unp.nominal_values(vE[vPeakInds])) ]),
						 colTitles=['col']*len(unp.nominal_values(vE[vPeakInds])),
						 rowTitles=["Peaks (keV)", "Element"],
						 mathMode=False)
	
	label = "Peaks: ["\
			+ ', '.join(['{:.2f}'.format(np.round(x,2))
						for i,x in enumerate(unp.nominal_values(vE[vPeakInds])) ])\
			+ "] keV"
	# label = ['\n'.join(wrap(x,20)) for x in label]
	
	ax[i].plot(unp.nominal_values(vE[vPeakInds]), vData[vPeakInds], 'rx', markersize=20,
			   label=label)
	ax[i].legend(loc='upper right')
	if not i % 2:
		ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.upper())

ax[-1].set_xlabel("Energy (keV)")
ax[-2].set_xlabel("Energy (keV)")
fig.subplots_adjust(hspace=0,wspace=0)
fig.savefig("Figures/" + "XRay-analysis")

print(vDetectedPeaks)
# pl.printAsLatexTable( np.array([['${:.0f}$'.format(x) for _,x in enumerate(vDetectedPeaks)],
# 								['element']*len(vDetectedPeaks) ]),
# 					colTitles=vANALYSIS_SAMPLE,
# 					rowTitles=["Peaks (keV)","Element"],
# 					mathMode=False )


#%% PUR and Voltage EFFECT COMPARISON
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()

for i, sample in enumerate(vCOMPARISON_SAMPLE):
	vData = np.genfromtxt(DATAPATH + FILE_PREFIX + "spek_" + sample + FILE_POSTFIX,
						  dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	vNoise = np.genfromtxt(DATAPATH + FILE_PREFIX + vEMPTY_MEAS[0] + FILE_POSTFIX,
						   dtype=float, delimiter='\n', skip_header=11, skip_footer=71, encoding='latin1')
	# vData -= vNoise
	vCh = np.arange(0, len(vData))
	vE = chToE(vCh)
	ax[i].semilogy(unp.nominal_values(vE), vData, 'b.', markersize=2)
	ax[i].set_xlabel("Energy (keV)")
	ax[i].set_ylabel("Event counts")
	ax[i].set_title(sample.replace("_"," ").upper())										# escape the underscore to calm down the latex interpreter

fig.savefig("Figures/" + "XRay-comparison")






plt.show()
# plt.close('all')








