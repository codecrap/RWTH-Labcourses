#-*- coding: utf-8 -*-
#
#@Geiger-counter_analysis.py:
#@author: Olexiy Fedorets
#@date: Thu 14.03.2019


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

# Characteristic curve (counts vs voltage)

vU, vCounts = np.genfromtxt(DATAPATH + "Geiger_characteristic_curve" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=3, unpack=True)
NOISE = 4																			# noise count was constant 4
vCounts -= NOISE

f_gauss = lambda p,x:  p[0] * np.exp(-(p[1]*x)**2) + p[2]
chifunc = lambda p,x,y,yerr: (y - f_gauss(p, x)) / yerr								# p[0] = d/dx line()
p0 = [1, 1, 1]

# lsqResults = spopt.least_squares(f_gauss, p0, args=(vU, vCounts), loss='linear', verbose=1)
# print(lsqResults.x,lsqResults.cost)
# vFitparam_stds = np.sqrt(np.diag(np.linalg.inv(lsqResults.jac.T.dot(lsqResults.jac)) ))

vFitparam,mCov,_,_,_ = spopt.leastsq(chifunc,p0,args=(vU,vCounts,np.sqrt(vCounts)),full_output=True)
chisq = np.sum(chifunc(vFitparam,vU,vCounts,np.sqrt(vCounts))**2) / (len(vCounts)-len(vFitparam))
vFitparam_std = np.sqrt(np.diag(mCov)*chisq)

pval = spst.chi2.sf(chisq,len(vCounts)-len(vFitparam))								# alternative (less accuracy for small pvals): 1 - spst.chi2.cdf(chisq,len(vCounts)-len(vFitparam))
pval_alt = spst.chisquare(vCounts,f_gauss(vFitparam,vU),ddof=len(vCounts)-len(vFitparam))

print(vFitparam,vFitparam_std,chisq,pval,pval_alt)

fig, ax = plt.subplots()
ax.plot(vU, vCounts, 'b.')
ax.plot(vU, f_gauss(vFitparam,vU), 'r-')
ax.set_title("Geiger counter characteristic curve")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Event counts")
fig.savefig("Figures/" + "Geiger_characteristic_curve")
fig.show()


# Gaussian distribution of counts verification

vCounts = np.genfromtxt(DATAPATH + "Geiger_gauss_1s_V2" + FILE_POSTFIX,
							dtype=float, delimiter='\n', skip_header=1, unpack=True)

vXaxis = np.linspace(min(vCounts),max(vCounts),len(vCounts))
mean,std = spst.norm.fit(vCounts)
vGaussfit = spst.norm.pdf(vXaxis,mean,std)

fig, ax = plt.subplots()
# generate counts of same type to test
# vCounts = np.random.normal(np.mean(vCounts),np.std(vCounts),1000)
ax.hist(vCounts, len(vCounts) , density=False, histtype='step')
ax.plot(vXaxis,vGaussfit,'r--',label=r"Gaussian fit: $\mu = %.1f, \sigma = %.1f$" % (mean,std))
# ax.plot(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),
# 		pl.gauss(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),np.mean(vCounts),np.std(vCounts), 1), 'r-')
ax.set_title("Gaussian distribution of counts for Geiger counter")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_gauss_histogram")
fig.show()


# Poisson distribution of counts for empty detector

# measured frequency of counts
vCounts = np.stack(np.full((1,60),0),np.full((1,34),1),np.full((1,17),2),axis=1)

f_poisson = lambda mu,k:  math.pow(mu,k) * np.exp(-mu) / math.factorial(k)
chifunc = lambda mu,k,y: (y - f_poisson(mu,k))										# p[0] = d/dx line()
mu0 = [np.mean(vCounts)]

vFitparam,mCov,_,_,_ = spopt.leastsq(chifunc,mu0,args=(np.unique(vCounts),vCounts),full_output=True)
chisq = np.sum(chifunc(vFitparam,np.unique(vCounts),vCounts)**2) / (len(vCounts)-len(vFitparam))
vFitparam_std = np.sqrt(np.diag(mCov)*chisq)

pval = spst.chi2.sf(chisq,len(vCounts)-len(vFitparam))								# alternative (less accuracy for small pvals): 1 - spst.chi2.cdf(chisq,len(vCounts)-len(vFitparam))
pval_alt = spst.chisquare(vCounts,f_gauss(vFitparam,vU),ddof=len(vCounts)-len(vFitparam))

print(vFitparam,vFitparam_std,chisq,pval,pval_alt)

fig, ax = plt.subplots()
# generate counts of same type to test
# vCounts = np.random.normal(np.mean(vCounts),np.std(vCounts),1000)
ax.hist(vCounts, len(vCounts) , normed=False)
ax.plot(np.linspace(0,10,100),f_poisson(vFitparam),'r--',label=r"Poisson fit: $\mu = %.1f$" % (vFitparam))
# ax.plot(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),
# 		pl.gauss(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),np.mean(vCounts),np.std(vCounts), 1), 'r-')
ax.set_title("Poisson distribution of counts for empty Geiger counter")
ax.set_xlabel("Number of events")
ax.set_ylabel("Event Frequency")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Geiger_poisson_histogram")
fig.show()