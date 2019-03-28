#-*- coding: utf-8 -*-
#
#@event-times_exp-dist.py:
#@author: Olexiy Fedorets
#@date: Wed 27.03.2019

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random
import scipy.stats as spst
import math
import peakutils as pu
import datetime as dt
import uncertainties.unumpy as unp

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

FILE_POSTFIX = ".CSV"
DATAPATH = "./Data/ALL0002/"

# compute current activity of samples used
# only Sr sample was used for geiger counter experiments
# half-life data from: https://www.nndc.bnl.gov/nudat2/reCenter.jsp?z=95&n=146

# define start values
vNames = ["Am","Sr","C"]
T_experiment = dt.datetime(2019, 3, 14, 12, 35, 17, 420514)
vT_start = [dt.datetime(1986,3,7),dt.datetime(1974,1,4),dt.datetime(2004,10,3)]
vActivity_start = np.array([3570,185,1]) * 10**3									# convert kBq -> Bq
vT_halflife = unp.uarray( [432.6,28.90,5700], [0.6,0.03,30] )						# in years

# compute activities at t=T_experiment
vDeltaT = np.array([ (T_experiment - t).total_seconds() for _,t in enumerate(vT_start) ])
vLambda =  np.log(2)/(vT_halflife *24*60*60*356)  									# convert years -> seconds
# print(vT_halflife,vLambda)
fActivity = lambda A0,l,t: A0 * unp.exp(-l*t)
vActivity_today = fActivity(vActivity_start,vLambda,vDeltaT)

print("Current activity:\n",vNames,"\n",vActivity_today)
pl.printAsLatexTable( np.array([ [x.strftime("%d.%m.%Y") for _,x in enumerate(vT_start)],
								['${:.0f}$'.format(x*10**-3) for _,x in enumerate(vActivity_start)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vT_halflife)],
								['${:.1ueL}$'.format(x) for _,x in enumerate(vLambda)],
								['${:.1ueL}$'.format(x*10**-3) for _,x in enumerate(vActivity_today)] ]),
					colTitles=vNames,
					rowTitles=["Buy date","Activity at buy date (kBq)","Halflife time (years)","Decay constant (1/s)","Activity today (kBq)"],
					mathMode=False )



vT, vU = np.genfromtxt(DATAPATH + "F0002CH1" + FILE_POSTFIX,
					dtype=float, delimiter=',', skip_header=18, usecols=(-3,-2), unpack=True)

# use very low voltage threshold to come most close to the counts counted by the external counter
vPeakInds = pu.peak.indexes(-vU, min_dist=0.01, thres=0.01, thres_abs=True)

fig, ax = plt.subplots()
ax.plot(vT*1e3,vU,'-',color='tab:orange')
ax.plot(vT[vPeakInds]*1e3,vU[vPeakInds],'bx',markersize=20)
ax.set_title("Geiger counter voltage"  )
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
fig.savefig("Figures/" + "Geiger_peaks_0,5s")
fig.show()

vDeltaTs = np.diff(vT[vPeakInds])*1e3
vHist,vBins = np.histogram(vDeltaTs, round(len(np.unique(vDeltaTs))/4), density=True)


# part I:
# use computed sample activity as estimate for expectation value of exponential distribution
vExpDistEstimate = spst.expon.pdf(vDeltaTs,scale=1/unp.nominal_values(vActivity_today[1]))

# part II:
# use mean as estimate for expectation value of exponential distribution
A = np.mean(vDeltaTs)/(vT[-1]-vT[0])
vExpDistEstimate = spst.expon.pdf(vDeltaTs,scale=1/A)

f_expdist = lambda A,dt: A * np.exp(-A*dt)
vExpDistEstimate = f_expdist(A,np.diff(vBins))

fig, ax = plt.subplots()
ax.hist(vDeltaTs, bins=round(len(np.unique(vDeltaTs))/4), density=True )
ax.plot(np.linspace(np.min(vDeltaTs),np.max(vDeltaTs),len(vExpDistEstimate)),vExpDistEstimate,'r-',label="Estimate using sample activity")
ax.set_title("Exponential distribution of event time differences")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Counts")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_eventtime_histogram")
fig.show()
