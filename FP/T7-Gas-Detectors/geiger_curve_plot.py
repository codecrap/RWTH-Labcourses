#-*- coding: utf-8 -*-
#
#@geiger_curve_plot.py:
#@author: Olexiy Fedorets
#@date: Thu 14.03.2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from numpy import random

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)


DATAPATH = "./Data/"
FILE_POSTFIX = ".txt"

vU, vCounts = np.genfromtxt(DATAPATH + "Geiger_characteristic_curve" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)
noise = 4
vCounts[1:] -= 4

line = lambda p,x: p[0] * np.exp(-(p[1]*x)**2) + p[2]

p0 = [1, 300, 0]  # start values
chifunc = lambda p, x, y: (y - line(p, x))   # p[0] = d/dx line()
fitparam, mCov, _, _, _ = spopt.leastsq(chifunc, p0, args=(vU, vCounts), full_output=True)
# print(fitparam,cov)
# chiq = np.sum(chifunc(fitparam, vU, np.zeros(vU.size), vCounts, np.sqrt(vCounts)) ** 2) / (len(vCounts) - len(fitparam))
# fitparam_err = np.sqrt(np.diag(mCov) * chiq)
# print(fitparam,fitparam_err,chiq)

fig, ax = plt.subplots()
ax.plot(vU, vCounts, 'b.')
ax.plot(vU, line(fitparam,vU), 'r-')
ax.set_title(r"Detector efficiency $\varepsilon$ vs energy")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$\varepsilon(E)$")
fig.savefig("Figures/" + "Efficiency")
fig.show()


vCounts = np.genfromtxt(DATAPATH + "Geiger_gauss_1s_V2" + FILE_POSTFIX,
							dtype=float, delimiter='\n', skip_header=1, unpack=True)

fig, ax = plt.subplots()
# vCounts = np.random.normal(np.mean(vCounts),np.std(vCounts),1000)
ax.hist(vCounts, 1000 )
ax.plot(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),
		np.max(vCounts) * pl.gauss(np.linspace(np.min(vCounts),np.max(vCounts),len(vCounts)),np.mean(vCounts),np.std(vCounts), 1/(np.sqrt(2*np.pi*np.std(vCounts)**2))), 'r-')
ax.set_title(r"Detector efficiency $\varepsilon$ vs energy")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"$\varepsilon(E)$")
fig.savefig("Figures/" + "Counts")
fig.show()