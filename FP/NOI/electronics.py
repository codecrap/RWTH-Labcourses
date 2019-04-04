#-*- coding: utf-8 -*-
#
#@electronics.py:
#@author: Olexiy Fedorets
#@date: Thu 04.04.2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import uncertainties.unumpy as unp
from uncertainties import ufloat

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl
from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

DATAPATH = "./Data/"
FILE_POSTFIX = ".txt"

vF, vVout = np.genfromtxt(DATAPATH + "Gain-vs-Frequency_Bandpassfilter_4b" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

vVout = unp.uarray(vVout,np.full(vVout.size,0.01))
vVin = ufloat(106,0.1)	# mV
vGain = vVout/vVin

# find bandwidth via -3dB (1/sqrt(2)) points
maxInd = np.where(vGain==np.max(vGain))[0][0]
print("Max gain: ",np.max(vGain),maxInd)
_,leftInd = pl.find_nearest(vGain[0:maxInd], np.max(vGain)/np.sqrt(2) )
_,rightInd = maxInd + pl.find_nearest(vGain[maxInd:-1], np.max(vGain)/np.sqrt(2) )
print("Bandpass left: ",vF[leftInd],leftInd," right: ",vF[rightInd],rightInd)

fig, ax = plt.subplots()
ax.errorbar(vF,unp.nominal_values(vGain),yerr=unp.std_devs(vGain),marker='o',color='b')
ax.axvline(vF[maxInd],color='k',linestyle='-',label="Max gain: ${:.1uL}$".format(vGain[maxInd]))
ax.axvline(vF[leftInd],color='r',linestyle='--',label="Highpass corner: %.1f kHz" % vF[leftInd])
ax.axvline(vF[rightInd],color='g',linestyle='--',label="Lowpass corner: %.1f kHz" % vF[rightInd])
ax.set_title("Bandpass filter frequency response, $f_1 = 1 \,\mathrm{kHz}, f_2 = 10 \,\mathrm{kHz}$")
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Gain")
ax.legend(loc='upper right')
fig.savefig("Figures/" + "Bandpass_GainVsFrequency_4b")