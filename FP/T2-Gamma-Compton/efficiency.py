#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Alexandre Drouet
#
# @efficiency.py: compute activity of radioactive sources after a given time,
# 					and use for detector efficiency calculation
# @author: Olexiy Fedorets
# @date: Tue 05.03.2019

import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import uncertainties.unumpy as unp
from uncertainties import ufloat
import operator
from functools import reduce
import peakutils as pu


import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


# Element order: Cs, Na, Co, Eu
vSOURCES = ['Cs', 'Na', 'Co', 'Eu']
DATAPATH = "./Data/"
FILE_POSTFIX = "_calibration.TKA"

# activity on day of experiment
# @activity.py: compute activity of radioactive sources after a given time
# @author: Olexiy Fedorets
# @date: Tue 19.02.2019
# define start values
# order: ["Cs (strong)","Cs (weak)","Na","Co","Eu"] consistent with calibration
T_experiment = dt.datetime(2019, 2, 19, 12, 35, 17, 420514)
vT_start = [dt.datetime(2010,11,23),dt.datetime(1988,8,12),
			dt.datetime(2005,1,12),dt.datetime(2003,4,15),dt.datetime(1978,6,2)]
vActivity_start = np.array([44400,37,37,37,37]) * 10**3 							# convert kBq -> Bq
vT_halflife = pl.uarray_tag( [11000,11000,950.5,1925.3,4943], [90,90,0.4,0.4,5], 'sys')	# in days

# compute activities at t=T_experiment
vDeltaT = np.array([ (T_experiment - t).total_seconds() for _,t in enumerate(vT_start) ])
vLambda =  np.log(2)/(vT_halflife *24*60*60)  										# convert days -> seconds
# print(vT_halflife,vLambda)
fActivity = lambda A0,l,t: A0 * unp.exp(-l*t)
vActivity_today = fActivity(vActivity_start,vLambda,vDeltaT)

print(vActivity_today)
pl.printAsLatexTable( np.array([ [x.strftime("%d.%m.%Y") for _,x in enumerate(vT_start)],
								['${:.0f}$'.format(x*10**-3) for _,x in enumerate(vActivity_start)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vT_halflife)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vLambda)],
								['${:.1ueL}$'.format(x*10**-3) for _,x in enumerate(vActivity_today)] ]),
					colTitles=["Cs (strong)","Cs (weak)","Na","Co","Eu"],
					rowTitles=["Buy date","Activity at buy date (\si{\kilo\becquerel})",
							   "Halflife time (days)","Decay constant (\si{\per\second})",
							   "Activity today (\si{\kilo\becquerel})"],
					mathMode=False )


# distance between source and detector
rS = ufloat(0.0875,0.01,'sys')														# 5+5*3/4, based on guess via picture and 5cm block length

# detector surface used
d_detector = ufloat(0.026,0.001,'sys')
F_detector = np.pi * (d_detector/2)**2


# photon yield (intensity)
vIntensity = [	[0.850],															# Cs
				[0.99940],															# Na
				[0.9985, 0.999826],													# Co
				[0.2858, 0.7580, 0.265, 0.1294, 0.1460, 0.210]	]					# Eu

# # peak bounds (manually set)
# Cs = [[420,475]]
# Na = [[820,865]]
# Co = [[750,800], [850,910]]
# Eu = [[85,105], [165,190], [225,255], [505,540], [620,665], [890,970]] #[690,780]
# vPeakbounds = [Cs, Na, Co, Eu]

# # peaks bounds old
# vPeakbounds = [[[400,490]],
#           [[810,890]],
#           [[740,810], [850,910]],
#           [[85,105], [160,195], [210,270], [480,560], [610,680], [690,780], [890,980]]]


vNoise =  np.genfromtxt(DATAPATH+"Noise"+FILE_POSTFIX, dtype=int, delimiter='\n', skip_header=2)
vCalibrationTheoryE,vCalibrationPeaks,_,vCalibrationSigmas,_ = np.loadtxt("photo_peaks.NORM")
vPeakFWHMs = pl.stdToFWHM(vCalibrationSigmas)

vEps = []
vPeakBounds = []
vM = []
# to match saved data in photo_peaks.NORM... : [Cs,Na,Co,Co,Eu,Eu,Eu,Eu,Eu,Eu]
vSOURCES_LIN = vSOURCES[0:2] + [vSOURCES[2]]*2 + [vSOURCES[3]]*6
vActivity_today_LIN = np.append(vActivity_today[1:3], [vActivity_today[3]]*2 + [vActivity_today[4]]*6)

for i,intensity in enumerate(reduce(operator.concat,vIntensity)):					# flatten the 2D list to match saved data in photo_peaks.NORM
	vData = np.genfromtxt(DATAPATH + vSOURCES_LIN[i] + FILE_POSTFIX, dtype=float, delimiter='\n', skip_header=2)
	vData -=  vNoise
	vBaseline = pu.baseline(vData,deg=8,max_it=200,tol=1e-4)
	vData -= vBaseline
	
	# get counts in peak
	vPeakBounds += [[int(np.rint(vCalibrationPeaks[i] - vPeakFWHMs[i])), int(np.rint(vCalibrationPeaks[i] + vPeakFWHMs[i]))]]
	_,vPeakCounts,_ = np.split(vData, vPeakBounds[i])
	vM += [np.sum(vPeakCounts)]
	
	# calc efficiency
	vEps += [ 4*np.pi * rS**2 * vM[i] / (vActivity_today_LIN[i] * intensity * F_detector) ]
	# deff = 4*np.pi*m/valI * np.sqrt((2*r*dr/(valA*F))**2 + (r**2*dA[i]/(valA**2*F))**2 + (r**2*dF/(valA)*F**2)**2)
		


print(vEps)
print("Efficiency mean: {:.2f}+-{:.2f}".format(*pl.weightedMean(unp.nominal_values(vEps),unp.std_devs(vEps))) )


fig, ax = plt.subplots()
ax.errorbar(vCalibrationTheoryE, unp.nominal_values(vEps), yerr=unp.std_devs(vEps), fmt='s', color='b')
ax.set_title(r"Detector efficiency $\varepsilon$ vs energy")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$\varepsilon(E)$")
fig.savefig("Figures/" + "Efficiency")
fig.show()

