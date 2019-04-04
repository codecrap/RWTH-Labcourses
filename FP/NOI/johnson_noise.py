#-*- coding: utf-8 -*-
#
#@johnson_noise.py:
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

def outputToNoise(V_sq,gain)

# compute actual gains
######################

vSetGain, vVout,  vVin = np.genfromtxt(DATAPATH + "Actual_Gain_Voltages_1a" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

# convert mV -> V
vVout = unp.uarray(vVout, np.full(vVout.size,0.01)) * 1e-3
vVin = unp.uarray(vVin, np.full(vVin.size,0.0001)) * 1e-3
vActualGain = vVout / vVin

pl.printAsLatexTable( np.transpose(np.array([ ['${:.1f}$'.format(x) for _,x in enumerate(vSetGain)],
												['${:.1ufL}$'.format(x) for _,x in enumerate(vActualGain)] ])),
					colTitles=["Set gain","Measured gain"],
					mathMode=False )


# analyse Johnson's noise at RT
###############################
LOWLEVEL_GAIN = 51 * 100
BANDWIDTH = 10e3 - 1e3
T = ufloat(294.4,0.1)

vR,_,_,_,vGain3,_,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsR_atRT" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for i,x in enumerate(vGain3)])

# convert to <V_J^2 + V_N^2>
vNoise = vVsq*10/((LOWLEVEL_GAIN*vGain3Actual)**2)

# linear fit
OMIT_VALS = -1
f_line = lambda p,x: p[0] * x + p[1]
chifunc = lambda p,x,xerr,y,yerr: (y - f_line(p,x)) / np.sqrt(yerr**2 + (p[0] * xerr)**2)
p0 = [1, 1]
vFitparam1,mCov,_,_,_ = spopt.leastsq(chifunc, p0,
										 args=(vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vNoise[0:OMIT_VALS]), unp.std_devs(vNoise[0:OMIT_VALS])),
										 full_output=True)
chiq1 = np.sum( chifunc(vFitparam1, vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vNoise[0:OMIT_VALS]), unp.std_devs(vNoise[0:OMIT_VALS]))**2 ) / (len(vNoise[0:OMIT_VALS]) - len(vFitparam1))
vFiterror1 = np.sqrt(np.diag(mCov) * chiq1)

V_N = ufloat(vFitparam1[-1],vFiterror1[-1])
vNoise_corrected = vNoise - V_N

# fit corrected noise
vFitparam2,mCov,_,_,_ = spopt.leastsq(chifunc, p0,
									args=(vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vNoise_corrected[0:OMIT_VALS]), unp.std_devs(vNoise_corrected[0:OMIT_VALS])),
									full_output=True)
chiq2 = np.sum( chifunc(vFitparam2, vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vNoise_corrected[0:OMIT_VALS]), unp.std_devs(vNoise_corrected[0:OMIT_VALS]))**2 ) / (len(vNoise_corrected[0:OMIT_VALS]) - len(vFitparam2))
vFiterror2 = np.sqrt(np.diag(mCov) * chiq2)

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(vR,unp.nominal_values(vNoise),yerr=unp.std_devs(vNoise),
			   label=r"$\left\langle V_J^2 + V_N^2 \right\rangle $ ",
			   marker='x',color='b',linestyle='none')
ax[0].plot(vR,f_line(vFitparam1,vR),'r-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparam1,vFiterror1)) )
ax[0].plot(vR,f_line(vFitparam2,vR),'m-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparam2,vFiterror2)) )
ax[0].errorbar(vR,unp.nominal_values(vNoise_corrected),yerr=unp.std_devs(vNoise_corrected),
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x',color='g',linestyle='none')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Johnson noise vs resistance")
ax[0].set_xlabel("Resistance $(\Omega)$")
ax[0].set_ylabel(r"$\left\langle V_J^2 + V_N^2 \right\rangle \,\, (\mathrm{V}^2)$ ")
ax[0].legend(loc='upper left')

ax[1].errorbar(vR[0:OMIT_VALS],unp.nominal_values(vNoise[0:OMIT_VALS]-f_line(vFitparam1,vR[0:OMIT_VALS])),
			   yerr=unp.std_devs(vNoise[0:OMIT_VALS]-f_line(vFitparam1,vR[0:OMIT_VALS])),marker='x',color='b',linestyle='none')
ax[1].plot(vR[0:OMIT_VALS],np.full(vR[0:OMIT_VALS].size,0),color='r',
			  label="$\chi^2/ndf = %.2f$" % chiq1)
ax[1].errorbar(vR[0:OMIT_VALS],unp.nominal_values(vNoise[0:OMIT_VALS]-f_line(vFitparam2,vR[0:OMIT_VALS])),
			   yerr=unp.std_devs(vNoise[0:OMIT_VALS]-f_line(vFitparam2,vR[0:OMIT_VALS])),marker='x',color='g',linestyle='none')
ax[1].plot(vR[0:OMIT_VALS],f_line(vFitparam1,vR[0:OMIT_VALS])-f_line(vFitparam2,vR[0:OMIT_VALS]),
			  color='m',label="$\chi^2/ndf = %.2f$" % chiq2)
ax[1].set_xscale('log',nonposx='clip')
ax[1].set_xlabel("Resistance $(\Omega)$")
ax[1].set_ylabel(r"$\left\langle V_J^2 + V_N^2 \right\rangle - (a \cdot R + b)$ ")
ax[1].legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsResistance")


# find k_B
k_B = ufloat(vFitparam2[0],vFiterror2[0]) / (4*T*BANDWIDTH)
print(k_B,k_B.std_score(1.3806503e-23))


# analyse Johnson's noise vs bandwidth
###############################
