#-*- coding: utf-8 -*-
#
#@auswertung_serienschwingkreis.py:
#@author: Olexiy Fedorets
#@date: Sun 07.01.2018


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import PraktLib as pl
import scipy.optimize as spopt
import scipy


def sigmaOmega(T,N):
	T = np.array(T)
	N = np.array(N)
	sigma_t = 10 * 10**(-6) #s
	omega = 2*np.pi/T
	sigma_omega = omega * (1/T) * np.sqrt(2)*sigma_t/N

	# val, err =  np.average(omega,weights=sigma_omega,returned=True)
	# return val,np.sqrt(err)
	return pl.weightedMean(omega,sigma_omega)

# Spannungsmessung:
# manual:
man_T_R1 = [0.00064,0.00057,0.00063]
man_N_R1 = [9,9,9]
man_T_R5 = [0.00064,0.00067,0.00067]
man_N_R5 = [3,3,2]
man_T_R10 = [0.00064,0.00068,0.00067]
man_N_R10 = [2,4,2]
man_T_R15 = [0.00072,0.00071,0.00075]
man_N_R15 = [1,1,2]

man_omegas = []
for Ti,Ni in zip([man_T_R1,man_T_R5,man_T_R10,man_T_R15],[man_N_R1,man_N_R5,man_N_R10,man_N_R15]):
	man_omegas.append(sigmaOmega(Ti,Ni))
print("manuelle frequenzen:\n",man_omegas)

# fft:
fft_R1 = np.array([1563,1547,1577])*2*np.pi
fft_R5 = np.array([1647,1689,1639])*2*np.pi
fft_R10 = np.array([1672,1625,1612])*2*np.pi
fft_R15 = np.array([1673,1669,1639])*2*np.pi

fft_omegas = []
fft_sigmas = []
for l in [fft_R1,fft_R5,fft_R10,fft_R15]:
	fft_omegas.append(np.mean(l))
	fft_sigmas.append(np.std(l)/np.sqrt(len(l)))
print("fft frequenzen:\n",fft_omegas,fft_sigmas)

# Strommessung:
# I_T_R1 = [0.00064,0.00066,0.00065]
# I_N_R1 = [6,5,5]
# I_T_R5 = [0.00067,0.00067,0.00065]
# I_N_R5 = [2,2,2]
# I_T_R10 = [0.00081,0.00068,0.00065]
# I_N_R10 = [1,1,2]
#
# I_omegas = []
# for Ti,Ni in zip([I_T_R1,I_T_R5,I_T_R10],[I_N_R1,I_N_R5,I_N_R10]):
# 	I_omegas.append(sigmaOmega(Ti,Ni))
# print(I_omegas)


# Dämpfungskonstanten:
# 1 Verfahren:
V1_R1 = np.array([0.56,0.55,0.59])*10**3
err_R1 = np.array([0.035,0.037,0.037])
V1_R5 = np.array([1.39,1.39,1.43])*10**3
err_R5 = np.array([0.073,0.071,0.081])
V1_R10 = np.array([2.42,2.38,2.17])*10**3
err_R10 = np.array([0.237,0.219,0.212])
V1_R15 = np.array([2.88,2.16,2.9])*10**3
err_R15 = np.array([0.645,0.44,0.646])

# 2 Verfahren:
V2_R1 = np.array([0.51,0.51,0.54])*10**3
V2_R5 = np.array([1.39,1.35,1.67])*10**3
V2_R10 = np.array([2.5,2.5,2.56])*10**3

Rs = np.array([1.1,5.2,10.1,15.3])

DeltaMeans1 = []
DeltaErrs1 = []
for l in [V1_R1,V1_R5,V1_R10,V1_R15]:
	DeltaMeans1.append(np.mean(l))
	DeltaErrs1.append(np.std(l)/np.sqrt(len(l)))

# DeltaMeans1 = []
# DeltaErrs1 = []
# for l,r in zip([V1_R1,V1_R5,V1_R10,V1_R15],[err_R1,err_R5,err_R10,err_R15]):
# 	DeltaMeans1.append(pl.weightedMean(l,r)[0])
# 	DeltaErrs1.append(pl.weightedMean(l,r)[1])

DeltaMeans2 = []
DeltaErrs2 = []
for l in [V2_R1,V2_R5,V2_R10]:
	DeltaMeans2.append(np.mean(l))
	DeltaErrs2.append(np.std(l)/np.sqrt(len(l)))

print("Deltas:\n",DeltaMeans1,DeltaErrs1,DeltaMeans2,DeltaErrs2)

fitparam,fitparam_err,chiq = pl.plotFit(Rs,np.zeros(4),np.array(DeltaMeans1),np.array(DeltaErrs1),
			title=r"Bestimmung von L und $R_L$ - Methode 1",xlabel=r"$R\; [\Omega]$",ylabel=r"$Dämpfung \;\delta \;[1/s]$")
print(fitparam,fitparam_err,chiq)
# L = 1/(2*a), sigma_L = sigma_a/(2*a^2)
# RL = x_0 = -b/a, sigma_RL = x_0 * sqrt((sigma_a/a)^2+(sigma_b/b)^2)
L1 = 1/(2*fitparam[0])
sigma_L1 = fitparam_err[0]/(2*fitparam[0]**2)
RL1 = fitparam[1]/fitparam[0]
sigma_RL1 = RL1 * np.sqrt((fitparam_err[0]/fitparam[0])**2 + (fitparam_err[1]/fitparam[1])**2)
print("L1 = %.3f \pm %.3f mH" % (L1*10**3,sigma_L1*10**3) )
print("RL1 = %.3f \pm %.3f Ohm" % (RL1,sigma_RL1) )


fitparam,fitparam_err,chiq = pl.plotFit(Rs[0:3],np.zeros(3),np.array(DeltaMeans2),np.array(DeltaErrs2),
			title=r"Bestimmung von L und $R_L$ - Methode 2",xlabel=r"$R\; [\Omega]$",ylabel=r"$Dämpfung \;\delta \;[1/s]$")
print(fitparam,fitparam_err,chiq)
L2 = 1/(2*fitparam[0])
sigma_L2 = fitparam_err[0]/(2*fitparam[0]**2)
RL2 = fitparam[1]/fitparam[0]
sigma_RL2 = RL2 * np.sqrt((fitparam_err[0]/fitparam[0])**2 + (fitparam_err[1]/fitparam[1])**2)
print("L2 = %.3f \pm %.3f mH" % (L2*10**3,sigma_L2*10**3) )
print("RL2 = %.3f \pm %.3f" % (RL2,sigma_RL2) )


# C Bestimmung:
def plotFit_modified(x,xerr,y,yerr,title="test",xlabel="",ylabel="",res_ylabel=r"$y - (-1 \cdot x + b)$",capsize=3,fontsize=20,show=True,method='leastsq'):
	def line(B, x):
		return -1*x + B[0]
	if method=='leastsq':
		p0 = [0]	# start values
		chifunc = lambda p,x,xerr,y,yerr: (y-line(p,x))/np.sqrt(yerr**2+p[0]*xerr**2)	# p[0] = d/dx line()
		fitparam,cov,_,_,_ = spopt.leastsq(chifunc,p0,args=(x,xerr,y,yerr),full_output=True)
		# print(fitparam,cov)
		chiq = np.sum(chifunc(fitparam,x,xerr,y,yerr)**2) / (len(y)-len(fitparam))
		fitparam_err = np.sqrt(np.diag(cov)*chiq)									# leastsq returns the 'fractional covariance matrix'
		# print(chiq,fitparam_err)

	if method=='ODR':
		_,_,b_ini,eb_ini,chiq_ini,corr_ini = pl.linreg_manual(x,y,yerr)

		model  = scipy.odr.Model(line)
		data   = scipy.odr.RealData(x, y, sx=xerr, sy=yerr)
		odr    = scipy.odr.ODR(data, model, beta0=[b_ini])
		output = odr.run()
		ndof = len(x)-2
		chiq = output.res_var*ndof
		# corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])

		fitparam = [output.beta[0]]
		fitparam_err = [output.sd_beta[0]]

	if show:
		fig,ax = plt.subplots(2,1,figsize=(15,10))
		residue = y-line(fitparam,x)
		ax[0].plot(x,line(fitparam,x),'r-',
					label="Fit: $b = %.3f \pm %.3f$"
							% (fitparam[0],fitparam_err[0]))
		# print("x:",x)
		# ax[0].plot(np.arange(1,5),(-1)*np.arange(1,5)*10**8+10**8,'k-')
		ax[0].errorbar(x,y,xerr=xerr,yerr=yerr,fmt='.',color='b',capsize=capsize-1)
		ax[0].set_title(title,fontsize=fontsize)
		ax[0].set_xlabel(xlabel,fontsize=fontsize)
		ax[0].set_ylabel(ylabel,fontsize=fontsize)
		ax[0].legend(loc='lower right',fontsize=fontsize)
		ax[0].grid(True)
		ax[1].errorbar(x,residue,yerr=np.sqrt(yerr**2+fitparam[0]*xerr**2),fmt='x',color='b',capsize=capsize,
					label=r"$\frac{\chi^2}{ndf} = %.E$" % np.around(chiq,3))
		ax[1].axhline(0,color='r')
		ax[1].set_title("Residuenverteilung",fontsize=fontsize)
		ax[1].set_xlabel(xlabel,fontsize=fontsize)
		ax[1].set_ylabel(res_ylabel,fontsize=fontsize)
		ax[1].legend(loc='upper right',fontsize=fontsize)
		ax[1].grid(True)
		fig.tight_layout()
		fig.savefig("Plots/"+title+".pdf",format='pdf',dpi=256)

	return fitparam,fitparam_err,chiq

pl.separator(20)
# print(len(DeltaMeans1),len(DeltaMeans2),len(DeltaErrs1),len(DeltaErrs2),len(man_omegas[0]),len(fft_omegas),len(fft_sigmas))

fitparam,fitparam_err,chiq = plotFit_modified(np.array(DeltaMeans1)**2,2*np.array(DeltaMeans1)*np.array(DeltaErrs1),
											np.array(list(map(lambda x: x[0]**2,man_omegas))), np.array(list(map(lambda x: x[1]*2*x[0],man_omegas))),
											title=r"Bestimmung von C - mit $\delta_1$ und $\omega_1$",xlabel=r"$\delta^2 \;[1/s^2]$",ylabel=r"$\omega^2 \;[1/s^2]$")
print(fitparam,fitparam_err,chiq)
# C = 1/(L*b), sigma_C = C * sqrt((sigma_L/L)^2+(sigma_b/b)^2)
C1 = 1/(L2*fitparam[0])
sigma_C1 = C1 * np.sqrt((sigma_L2/L2)**2+(fitparam_err[0]/fitparam)**2)
print("C1 = %.3f \pm %.3f" % (C1*10**6,sigma_C1*10**6) )

fitparam,fitparam_err,chiq = plotFit_modified(np.array(DeltaMeans2)**2,2*np.array(DeltaMeans2)*np.array(DeltaErrs2),
											np.array(list(map(lambda x: x[0]**2,man_omegas[0:3]))), np.array(list(map(lambda x: x[1]*2*x[0],man_omegas[0:3]))),
											title=r"Bestimmung von C - mit $\delta_2$ und $\omega_1$",xlabel=r"$\delta^2 \;[1/s^2]$",ylabel=r"$\omega^2 \;[1/s^2]$")
print(fitparam,fitparam_err,chiq)
# C = 1/(L*b), sigma_C = C * sqrt((sigma_L/L)^2+(sigma_b/b)^2)
C1 = 1/(L2*fitparam[0])
sigma_C1 = C1 * np.sqrt((sigma_L2/L2)**2+(fitparam_err[0]/fitparam)**2)
print("C1 = %.3f \pm %.3f" % (C1*10**6,sigma_C1*10**6) )


fitparam,fitparam_err,chiq = plotFit_modified(np.array(DeltaMeans1)**2,2*np.array(DeltaMeans1)*np.array(DeltaErrs1),
											np.array(fft_omegas)**2, np.array(fft_omegas)*np.array(fft_sigmas)*2,
											title=r"Bestimmung von C - mit $\delta_1$ und $\omega_2$",xlabel=r"$\delta^2 \;[1/s^2]$",ylabel=r"$\omega^2 \;[1/s^2]$")
print(fitparam,fitparam_err,chiq)
# C = 1/(L*b), sigma_C = C * sqrt((sigma_L/L)^2+(sigma_b/b)^2)
C1 = 1/(L2*fitparam[0])
sigma_C1 = C1 * np.sqrt((sigma_L2/L2)**2+(fitparam_err[0]/fitparam)**2)
print("C1 = %.3f \pm %.3f" % (C1*10**6,sigma_C1*10**6) )

fitparam,fitparam_err,chiq = plotFit_modified(np.array(DeltaMeans2)**2,2*np.array(DeltaMeans2)*np.array(DeltaErrs2),
											np.array(fft_omegas[0:3])**2, np.array(fft_omegas[0:3])*np.array(fft_sigmas[0:3])*2,
											title=r"Bestimmung von C - mit $\delta_2$ und $\omega_2$",xlabel=r"$\delta^2 \;[1/s^2]$",ylabel=r"$\omega^2 \;[1/s^2]$")
print(fitparam,fitparam_err,chiq)
# C = 1/(L*b), sigma_C = C * sqrt((sigma_L/L)^2+(sigma_b/b)^2)
C1 = 1/(L2*fitparam[0])
sigma_C1 = C1 * np.sqrt((sigma_L2/L2)**2+(fitparam_err[0]/fitparam)**2)
print("C1 = %.3f \pm %.3f" % (C1*10**6,sigma_C1*10**6) )




# fitparam,fitparam_err,chiq = plotFit_modified(np.array(DeltaMeans2)**2,2*np.array(DeltaMeans2)*np.array(DeltaErrs2),
# 											np.array(list(map(lambda x: x[0]**2,I_omegas))), np.array(list(map(lambda x: x[1]*2*x[0],I_omegas))),
# 			title=r"Bestimmung von C - Frequenzen aus I-Messung",xlabel=r"$\delta^2 \;[1/s^2]$",ylabel=r"$\omega^2 \;[1/s^2]$")
# print(fitparam,fitparam_err,chiq)
# # C = 1/(L*b), sigma_C = C * sqrt((sigma_L/L)^2+(sigma_b/b)^2)
# C2 = 1/(L2*fitparam[0])
# sigma_C2 = C2 * np.sqrt((sigma_L2/L2)**2+(fitparam_err[0]/fitparam)**2)
# print("C2 = %.3f \pm %.3f" % (C2,sigma_C2) )






plt.show()
