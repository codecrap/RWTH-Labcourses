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

def outputVToNoiseV(V_sq, gain):
	return V_sq*10/(gain**2)

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
BANDWIDTH = 10e3 - 1e3 # Hz
T_RT = ufloat(294.4, 0.1) # K
K_B_TRUE = 1.3806503e-23

vR,_,_,_,vGain3,_,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsR_atRT" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for _,x in enumerate(vGain3)])

# convert to <V_J^2 + V_N^2>
vNoise = outputVToNoiseV(vVsq, LOWLEVEL_GAIN * vGain3Actual)

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

V_N2 = ufloat(vFitparam1[-1], vFiterror1[-1])
vV_J2 = vNoise - V_N2

# fit corrected noise
vFitparam2,mCov,_,_,_ = spopt.leastsq(chifunc, p0,
									  args=(vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])),
									  full_output=True)
chiq2 = np.sum(chifunc(vFitparam2, vR[0:OMIT_VALS], np.zeros(vR[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])) ** 2) / (len(vV_J2[0:OMIT_VALS]) - len(vFitparam2))
vFiterror2 = np.sqrt(np.diag(mCov) * chiq2)

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(vR,unp.nominal_values(vNoise),yerr=unp.std_devs(vNoise),
			   label=r"$\left\langle V_J^2 + V_N^2 \right\rangle $ ",
			   marker='x',color='g',linestyle='none')
ax[0].plot(vR,f_line(vFitparam1,vR),'g-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparam1,vFiterror1)) )
ax[0].errorbar(vR, unp.nominal_values(vV_J2), yerr=unp.std_devs(vV_J2),
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x', color='m', linestyle='none')
ax[0].plot(vR,f_line(vFitparam2,vR),'m-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparam2,vFiterror2)) )
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Johnson noise vs resistance")
ax[0].set_xlabel("Resistance $(\Omega)$")
ax[0].set_ylabel(r"$\left\langle V^2 \right\rangle \quad\quad (\mathrm{V}^2)$ ")
ax[0].legend(loc='upper left')

ax[1].errorbar(vR[0:OMIT_VALS],unp.nominal_values(vNoise[0:OMIT_VALS]-f_line(vFitparam1,vR[0:OMIT_VALS])),
			   yerr=unp.std_devs(vNoise[0:OMIT_VALS]-f_line(vFitparam1,vR[0:OMIT_VALS])),marker='x',color='g',linestyle='none')
ax[1].plot(vR[0:OMIT_VALS],np.full(vR[0:OMIT_VALS].size,0),color='g',
			  label="$\chi^2/ndf = %.2f$" % chiq1)
ax[1].errorbar(vR[0:OMIT_VALS],unp.nominal_values(vNoise[0:OMIT_VALS]-f_line(vFitparam2,vR[0:OMIT_VALS])),
			   yerr=unp.std_devs(vNoise[0:OMIT_VALS]-f_line(vFitparam2,vR[0:OMIT_VALS])),marker='x',color='m',linestyle='none')
ax[1].plot(vR[0:OMIT_VALS],f_line(vFitparam1,vR[0:OMIT_VALS])-f_line(vFitparam2,vR[0:OMIT_VALS]),
			  color='m',label="$\chi^2/ndf = %.2f$" % chiq2)
ax[1].set_xscale('log',nonposx='clip')
ax[1].set_xlabel("Resistance $(\Omega)$")
ax[1].set_ylabel(r"$\left\langle V^2 \right\rangle - (a \cdot R + b) \quad\quad (\mathrm{V}^2)$ ")
ax[1].legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsResistance")


# find k_B
k_B = ufloat(vFitparam2[0],vFiterror2[0]) / (4 * T_RT * BANDWIDTH)
print(k_B,k_B.std_score(K_B_TRUE))


# analyse Johnson's noise vs bandwidth @RT
###########################################
R_IN = 10e3 #10 kOhm resistor
# ENBW from manual, should correspond to set f1 and f2
vENBW = np.array([248,355,784,1077,3554,9997,10774,33324,35543,107740,110961,111061])
vENBW_ERRORS = vENBW * 0.04

# amplifier noise
vF1,vF2,vGain3,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsBW_atRT_AmplifierV-R10Ohm" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

vBW = vF1 - vF2
# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for _,x in enumerate(vGain3)])

# convert to <V_N^2>
vV_N2 = outputVToNoiseV(vVsq, LOWLEVEL_GAIN * vGain3Actual)

# resistance johnson's noise
vF1,vF2,vGain3,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsBW_atRT_JohnsonV-R10kOhm" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

vBW = vF1 - vF2
# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for _,x in enumerate(vGain3)])

# convert to <V_J^2 + V_N^2>
vNoise = outputVToNoiseV(vVsq, LOWLEVEL_GAIN * vGain3Actual)
# convert to <V_J^2>
vV_J2 = vNoise - vV_N2


# linear fit
OMIT_VALS = len(vBW)
p0 = [1, 1]
vFitparamBW,mCov,_,_,_ = spopt.leastsq(chifunc, p0,
									  args=(vBW[0:OMIT_VALS], np.zeros(vBW[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])),
									  full_output=True)
chiqBW = np.sum(chifunc(vFitparamBW, vBW[0:OMIT_VALS], np.zeros(vBW[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])) ** 2) / (len(vV_J2[0:OMIT_VALS]) - len(vFitparamBW))
vFiterrorBW = np.sqrt(np.diag(mCov) * chiqBW)

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(vBW, unp.nominal_values(vV_J2), yerr=unp.std_devs(vV_J2),
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x', color='m', linestyle='none')
ax[0].plot(vBW,f_line(vFitparamBW,vBW),'m-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparamBW,vFiterrorBW)) )
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Johnson noise vs bandwidth at RT")
ax[0].set_xlabel("Bandwidth $\Delta f$ (Hz)")
ax[0].set_ylabel(r"$\left\langle V_J^2 \right\rangle \quad\quad (\mathrm{V}^2)$ ")
ax[0].legend(loc='upper left')

ax[1].errorbar(vBW[0:OMIT_VALS],unp.nominal_values(vV_J2[0:OMIT_VALS]-f_line(vFitparamBW,vBW[0:OMIT_VALS])),
			   yerr=unp.std_devs(vV_J2[0:OMIT_VALS]-f_line(vFitparamBW,vBW[0:OMIT_VALS])),marker='x',color='m',linestyle='none')
ax[1].axhline(0,color='m',label="$\chi^2/ndf = %.2f$" % chiqBW)
ax[1].set_xscale('log',nonposx='clip')
ax[1].set_xlabel("Bandwidth $\Delta f$  (Hz)")
ax[1].set_ylabel(r"$\left\langle V_J^2\right\rangle - (a \cdot \Delta f + b) \quad\quad (\mathrm{V}^2)$ ")
ax[1].legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsBandwidth_RT")

# find k_B
k_B = ufloat(vFitparamBW[0],vFiterrorBW[0]) / (4 * T_RT * R_IN)
print(k_B,k_B.std_score(K_B_TRUE))

# plot vs ENBW
fig, ax = plt.subplots()

ax.errorbar(vENBW, unp.nominal_values(vV_J2), yerr=unp.std_devs(vV_J2), xerr=vENBW_ERRORS,
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x', color='m', linestyle='-')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Johnson noise vs equivalent noise bandwidth at RT")
ax.set_xlabel("ENBW $\Delta f$ (Hz)")
ax.set_ylabel(r"$\left\langle V_J^2 \right\rangle \quad (\mathrm{V}^2)$ ")
ax.legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsENBW_RT")

# plot noise power spectral density
fig, ax = plt.subplots()

ax.errorbar(vBW, unp.nominal_values(vV_J2/vBW), yerr=unp.std_devs(vV_J2/vBW),
			   label=r"$\left\langle V_J^2 \right\rangle / \Delta f $ ",
			   marker='x', color='b', linestyle='-')
# ax.axhline(pl.weightedMean(unp.nominal_values(vV_J2/vBW),unp.std_devs(vV_J2/vBW))[0],color='r',label=r"$4 k_B T_{RT} R_B$")
ax.axhline(unp.nominal_values(4*k_B*T_RT*R_IN),color='r',label=r"$4 k_{B,fit} T_{RT} R_B$")
ax.axhline(unp.nominal_values(4*K_B_TRUE*T_RT*R_IN),color='k',label=r"$4 k_{B,true} T_{RT} R_B$")
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_title("Johnson noise power spectral density")
ax.set_xlabel("Bandwidth $\Delta f$ (Hz)")
ax.set_ylabel(r"$\left\langle V_J^2 \right\rangle / \Delta f \quad (\mathrm{V}^2/\mathrm{Hz})$ ")
ax.legend(loc='lower left')

fig.savefig("Figures/" + "Johnson-NoisePSD_RT")



# analyse Johnson's noise vs bandwidth @77K
###########################################
R_A = 9.7
R_B = 9.99 * 1e3
T_N2 = 77 # K

# amplifier noise
vF1,vF2,vGain3,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsBW_at77K_AmplifierV-R10Ohm" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

vBW = vF1 - vF2
# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for _,x in enumerate(vGain3)])

# convert to <V_N^2>
vV_N2 = outputVToNoiseV(vVsq, LOWLEVEL_GAIN * vGain3Actual)

# resistance johnson's noise
vF1,vF2,vGain3,vVsq1,vVsq2,vVsq3,_ = np.genfromtxt(DATAPATH + "Johnson-Noise_vsBW_at77K_JohnsonV-R10kOhm" + FILE_POSTFIX,
							dtype=float, delimiter=' ', skip_header=2, unpack=True)

vBW = vF1 - vF2
# convert mV -> V
vVsq = unp.uarray(np.mean([vVsq1,vVsq2,vVsq3],axis=0),
				  np.std([vVsq1,vVsq2,vVsq3],axis=0)/np.sqrt(3) ) * 1e-3
# in case of ambiguity, choose first option from set gains
vGain3Actual = np.array([vActualGain[vSetGain==x][0] for _,x in enumerate(vGain3)])

# convert to <V_J^2 + V_N^2>
vNoise = outputVToNoiseV(vVsq, LOWLEVEL_GAIN * vGain3Actual)
# convert to <V_J^2>
vV_J2 = vNoise - vV_N2


# linear fit
OMIT_VALS = len(vBW)
p0 = [1, 1]
vFitparamBW2,mCov,_,_,_ = spopt.leastsq(chifunc, p0,
									  args=(vBW[0:OMIT_VALS], np.zeros(vBW[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])),
									  full_output=True)
chiqBW2 = np.sum(chifunc(vFitparamBW2, vBW[0:OMIT_VALS], np.zeros(vBW[0:OMIT_VALS].size), unp.nominal_values(vV_J2[0:OMIT_VALS]), unp.std_devs(vV_J2[0:OMIT_VALS])) ** 2) / (len(vV_J2[0:OMIT_VALS]) - len(vFitparamBW2))
vFiterrorBW2 = np.sqrt(np.diag(mCov) * chiqBW2)

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(vBW, unp.nominal_values(vV_J2), yerr=unp.std_devs(vV_J2),
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x', color='m', linestyle='none')
ax[0].plot(vBW,f_line(vFitparamBW2,vBW),'m-',
		   label="$a = {:.1ueL}, b = {:.1ueL}$".format(*unp.uarray(vFitparamBW2,vFiterrorBW2)) )
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Johnson noise vs bandwidth at 77K")
ax[0].set_xlabel("Bandwidth $\Delta f$ (Hz)")
ax[0].set_ylabel(r"$\left\langle V_J^2 \right\rangle \quad\quad (\mathrm{V}^2)$ ")
ax[0].legend(loc='upper left')

ax[1].errorbar(vBW[0:OMIT_VALS],unp.nominal_values(vV_J2[0:OMIT_VALS]-f_line(vFitparamBW2,vBW[0:OMIT_VALS])),
			   yerr=unp.std_devs(vV_J2[0:OMIT_VALS]-f_line(vFitparamBW2,vBW[0:OMIT_VALS])),marker='x',color='m',linestyle='none')
ax[1].axhline(0,color='m',label="$\chi^2/ndf = %.2f$" % chiqBW2)
ax[1].set_xscale('log',nonposx='clip')
ax[1].set_xlabel("Bandwidth $\Delta f$  (Hz)")
ax[1].set_ylabel(r"$\left\langle V_J^2\right\rangle - (a \cdot \Delta f + b) \quad\quad (\mathrm{V}^2)$ ")
ax[1].legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsBandwidth_77K")

# find k_B
k_B = ufloat(vFitparamBW2[0],vFiterrorBW2[0]) / (4 * T_N2 * R_B)
print(k_B,k_B.std_score(K_B_TRUE))

# plot vs ENBW
fig, ax = plt.subplots()

ax.errorbar(vENBW, unp.nominal_values(vV_J2), yerr=unp.std_devs(vV_J2),  xerr=vENBW_ERRORS,
			   label=r"$\left\langle V_J^2 \right\rangle $ ",
			   marker='x', color='m', linestyle='-')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Johnson noise vs equivalent noise bandwidth at 77K")
ax.set_xlabel("ENBW $\Delta f$ (Hz)")
ax.set_ylabel(r"$\left\langle V_J^2 \right\rangle \quad (\mathrm{V}^2)$ ")
ax.legend(loc='upper left')

fig.savefig("Figures/" + "Johnson-NoiseVsENBW_77K")

# plot noise power spectral density
fig, ax = plt.subplots()

ax.errorbar(vBW, unp.nominal_values(vV_J2/vBW), yerr=unp.std_devs(vV_J2/vBW),
			   label=r"$\left\langle V_J^2 \right\rangle / \Delta f $ ",
			   marker='x', color='b', linestyle='-')
# ax.axhline(pl.weightedMean(unp.nominal_values(vV_J2/vBW),unp.std_devs(vV_J2/vBW))[0],color='r',label=r"$4 k_B T_{RT} R_B$")
ax.axhline(unp.nominal_values(4*k_B*T_N2*R_B),color='r',label=r"$4 k_{B,fit} T_{RT} R_B$")
ax.axhline(unp.nominal_values(4*K_B_TRUE*T_N2*R_B),color='k',label=r"$4 k_{B,true} T_{RT} R_B$")
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_title("Johnson noise power spectral density")
ax.set_xlabel("Bandwidth $\Delta f$ (Hz)")
ax.set_ylabel(r"$\left\langle V_J^2 \right\rangle / \Delta f \quad (\mathrm{V}^2/\mathrm{Hz})$ ")
ax.legend(loc='lower left')

fig.savefig("Figures/" + "Johnson-NoisePSD_77K")