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
from uncertainties import ufloat

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

FILE_POSTFIX = ".CSV"
DATAPATH = "./Data/ALL0002/"

# fraction of unique values to be used for bins
BIN_FRACTION = 6

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
vT_elapsed = np.array([(T_experiment - t).total_seconds() for _, t in enumerate(vT_start)])
vLambda =  np.log(2)/(vT_halflife *24*60*60*356)  									# convert years -> seconds
# print(vT_halflife,vLambda)
fActivity = lambda A0,l,t: A0 * unp.exp(-l*t)
vActivity_today = fActivity(vActivity_start, vLambda, vT_elapsed)

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
# fig.show()

vDeltaT = np.diff(vT[vPeakInds]) * 1e3												# convert s -> ms
vHist,vBins = np.histogram(vDeltaT, round(len(np.unique(vDeltaT)) / BIN_FRACTION), density=True)
vBinMiddle = vBins[0:-1]+np.diff(vBins)/2

fig, ax = plt.subplots()
ax.hist(vDeltaT, bins=round(len(np.unique(vDeltaT)) / BIN_FRACTION), density=False, edgecolor='k')
ax.set_title("Histogram of event time-difference frequency")
ax.set_xlabel("Time difference between two peaks $\Delta t$ (ms)")
ax.set_ylabel("True frequency")
fig.savefig("Figures/" + "Geiger_eventtime_histogram_unnormalized")
# fig.show()

vCounts = np.genfromtxt("./Data/Geiger_exponential_01s.txt",
					dtype=float, delimiter=',', skip_header=1, unpack=True)

vDeltaT01 = 0.1 / vCounts
fig, ax = plt.subplots()
ax.hist(vDeltaT01, bins=round(len(np.unique(vDeltaT01))/4), density=False, edgecolor='k')
ax.set_title("Histogram of event time-difference frequency")
ax.set_xlabel("Time difference between two peaks $\Delta t$ (ms)")
ax.set_ylabel("True frequency")
fig.savefig("Figures/" + "Geiger_eventtime_histogram_unnormalized_01s")

f_expdist = lambda A,dt: A * np.exp(-A*dt)

# part I:
# use computed sample activity as estimate for expectation value of exponential distribution
# vExpDist_activityEstimate = spst.expon.pdf(vDeltaT, scale=unp.nominal_values(vActivity_today[1]))
vExpDist_activityEstimate = f_expdist(unp.nominal_values(vActivity_today[1]),vBinMiddle)


# part II:
# use mean as estimate for expectation value of exponential distribution
# vExpDist_meanEstimate = spst.expon.pdf(vDeltaT, scale=1/A)
A = len(vPeakInds) / (vT[-1] - vT[0])
vExpDist_meanEstimate = f_expdist(A,vBinMiddle)


# part III:
# use least-squares fitting of distribution to the histogram data
chifunc = lambda A,dt,y: (y - f_expdist(A,dt))
vFitparam_start = [1]
vFitparam,mCov,_,_,_ = spopt.leastsq(chifunc,vFitparam_start,args=(vBinMiddle,vHist),full_output=True)

# exclude bins with count less than 5 from chisquared test
vHistTrue,vBinsTrue = np.histogram(vDeltaT, round(len(np.unique(vDeltaT)) / BIN_FRACTION), density=False)
vMask = vHistTrue > 5

chisq = np.sum(chifunc(vFitparam,vBinMiddle[vMask],vHist[vMask])**2) / (len(vBinMiddle[vMask])-len(vFitparam))
pval = spst.chi2.sf(chisq,len(vBinMiddle[vMask])-len(vFitparam))					# alternative (less accuracy for small pvals): 1 - spst.chi2.cdf(chisq,len(vCounts)-len(vFitparam))
print(chisq,pval)

chisq_alt,pval_alt = spst.chisquare(vHist[vMask],f_expdist(vFitparam,vBinMiddle[vMask]),ddof=len(vBinMiddle[vMask])-len(vFitparam))
pval_alt = spst.chi2.sf(chisq_alt,len(vBinMiddle[vMask])-len(vFitparam))

vFitparam_std = np.sqrt(np.diag(mCov)*chisq)
print(vFitparam,vFitparam_std)
print(chisq_alt,pval_alt)

# compute values needed for distribution characteristic
print("Expected values from observed event rate: mu,sigma = ", 1/A)
print("Expected values from fit: mu,sigma = ", 1/ufloat(vFitparam,vFitparam_std))

fig, ax = plt.subplots()
ax.hist(vDeltaT, bins=round(len(np.unique(vDeltaT)) / BIN_FRACTION), density=True, edgecolor='k')
ax.plot(vBinMiddle,f_expdist(vFitparam,vBinMiddle),'r-',
		label="Nonlinear least-squares fit,\n $\chi^2 = %.1e, \, p = %.1e$" % (chisq,pval))
ax.plot(vBinMiddle, vExpDist_activityEstimate, 'k-',
		label="Estimate using sample activity")
ax.plot(vBinMiddle, vExpDist_meanEstimate, 'g-',
		label="Estimate using observed event rate")
ax.set_title("Exponential distribution of event time differences")
ax.set_xlabel("Time difference between two peaks $\Delta t$ (ms)")
ax.set_ylabel("Normalized frequency")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_eventtime_histogram_fit")
# fig.show()


