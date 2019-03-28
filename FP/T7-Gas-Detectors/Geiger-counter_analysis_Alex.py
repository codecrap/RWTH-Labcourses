#-*- coding: utf-8 -*-
#
#@Geiger-counter_analysis.py:
#@author: Alex


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random
import scipy.stats as spst
import math

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


DATAPATH = "./Data/"
FILE_POSTFIX = ".txt"

# Gaussian distribution of counts verification
print("Gaussian distribution")
# get data
vCounts = np.genfromtxt(DATAPATH + "Geiger_gauss_1s_V2" + FILE_POSTFIX,
							dtype=float, delimiter='\n', skip_header=1, unpack=True)
vXaxis = np.linspace(min(vCounts), max(vCounts), np.unique(vCounts).size)

# get mean and std
mean = np.mean(vCounts)
std = np.std(vCounts, ddof=1)
meanstd = std/np.sqrt(len(vCounts))
print("mean = %.2f \pm %.2f" % (mean,meanstd))
print("std = %.2f" % std)

# get histogram array
vHist, _ = np.histogram(vCounts, np.unique(vCounts).size, density=True)
vHistReal, _ = np.histogram(vCounts, np.unique(vCounts).size, density=False)

# calc expected gauss
print("estimate")
vNormal = pl.gauss(vXaxis, mean, std, 1/(np.sqrt(2*np.pi)*std))

# chi-test
mask = vHistReal>4

#chiSq = np.sum((vHist[mask] - vNormal[mask])**2)
#pVal = spst.chi2.sf(chiSq, len(vHist[mask])-2-1)
#print("%.10f" % chiSq)
#print("%.10f" % pVal)

chiSq, pVal = spst.chisquare(vHist[mask], vNormal[mask], ddof=2)
print("chi2 = %.3f" % chiSq)
print("p = %.3f" % pVal)

# gauss fit
print("leastsq fit")
def gaussFunc(x,m,s):
	return pl.gauss(x, m, s, 1/(np.sqrt(2*np.pi)*s))

opt, cov = spopt.curve_fit(gaussFunc, vXaxis, vHist, [mean, std])
print("mean = %.2f \pm %.2f" % (opt[0],cov[0][0]))
print("std = %.2f \pm %.2f" % (opt[1],cov[1][1]))

gaussFit = gaussFunc(vXaxis, opt[0], opt[1])
chiSq, pVal = spst.chisquare(vHist[mask], gaussFit[mask], ddof=2)
print("chi2 = %.3f" % chiSq)
print("p = %.3f" % pVal)

# plot
fig, ax = plt.subplots()
ax.hist(vCounts, np.unique(vCounts).size, density=False)
ax.set_title("Counts for Geiger counter with Sr")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
#ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_gauss_histogram")
fig.show()

fig, ax = plt.subplots()
ax.hist(vCounts, np.unique(vCounts).size, density=True, histtype='step')
ax.plot(vXaxis, vNormal, 'r--', label=r"expected gaussian: $\mu = %.2f, \sigma = %.2f$" % (mean,std))
ax.plot(vXaxis, gaussFit, 'g--', label=r"fitted gaussian: $\mu = %.2f, \sigma = %.2f$" % (opt[0],opt[1]))
ax.set_title("Gaussian distribution of counts for Geiger counter with Sr")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_gauss_fit")
fig.show()


# Poisson distribution of counts in empty detector
print("Poisson distribution")

# poisson dist
def poisson(x, mu):
	return np.array([math.pow(mu,val) * np.exp(-val) / math.factorial(val) for val in x])

# get data
vCounts = np.concatenate([np.full(60,0), np.full(34,1), np.full(17,2), [3,4]])
vXaxis = np.linspace(min(vCounts), max(vCounts), np.unique(vCounts).size)
vXaxisLong = vXaxis

# get mean and std
mean = np.mean(vCounts)
std = np.sqrt(mean)
meanstd = std/np.sqrt(len(vCounts))
print("mean = %.2f \pm %.2f" % (mean,meanstd))
print("std = %.2f" % std)

# get histogram array
vHist, _ = np.histogram(vCounts, np.unique(vCounts).size, density=True)
vHistReal, _ = np.histogram(vCounts, np.unique(vCounts).size, density=False)

# calc expected poisson
print("estimate")
vPoisson = poisson(vXaxis, mean)
vPoissonLong = poisson(vXaxisLong, mean)

# chi-test
mask = vHistReal>4

chiSq, pVal = spst.chisquare(vHist[mask], vPoisson[mask], ddof=1)
print("chi2 = %.2f" % chiSq)
print("p = %.2f" % pVal)

# poisson fit
print("leastsq fit")

opt, cov = spopt.curve_fit(poisson, vXaxis, vHist, [mean])
print("mean = %.2f \pm %.2f" % (opt[0],cov[0][0]))
std = 0.5*cov[0][0]/np.sqrt(opt[0])
#print("std = %.2f \pm %.2f" % (opt[1],cov[1][1]))

poissonFit = poisson(vXaxis, opt[0])
poissonFitLong = poisson(vXaxisLong, opt[0])
chiSq, pVal = spst.chisquare(vHist[mask], poissonFit[mask], ddof=1)
print("chi2 = %.2f" % chiSq)
print("p = %.2f" % pVal)

# plot
fig, ax = plt.subplots()
ax.hist(vCounts, np.unique(vCounts).size, density=False)
ax.set_title("Counts for empty Geiger counter")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
#ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_poisson_histogram")
fig.show()

fig, ax = plt.subplots()
ax.hist(vCounts, np.unique(vCounts).size, density=True, histtype='step')
ax.plot(vXaxisLong, vPoissonLong, 'rx', label=r"expected poisson: $\mu = %.1f$" % (mean))
ax.plot(vXaxisLong, poissonFitLong, 'gx', label=r"fitted poisson: $\mu = %.1f$" % (opt[0]))
ax.set_title("Poisson distribution of counts for empty Geiger counter")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_poisson_fit")
fig.show()