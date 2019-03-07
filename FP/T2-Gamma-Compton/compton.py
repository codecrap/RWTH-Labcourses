#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:21:13 2019

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from pygments.lexers.testing import TAPLexer
from scipy.optimize import curve_fit
from calibration import ChtoE
from scipy.constants import c, m_e, e
import sys
sys.path.append("./../../")
import PraktLib as pl

import numpy as np
from matplotlib import pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit
from calibration import ChtoE
import scipy.constants as sc

import sys
sys.path.append("./../../")															# path needed for PraktLib
import PraktLib as pl

import matplotlib
matplotlib.style.use("../labreport.mplstyle")

from importlib import reload														# take care of changes in module by manually reloading
pl = reload(pl)

# computed by efficiency.py
EPS_EFF = ufloat(0.25,0.02,'sys')													# efficiency mean
A_CS = ufloat(36727880.177579954,57006.008076133214,'sys')
# theory values from datasheet
I_CS = 0.85
E0_CS = 661.66																		# in keV, energy of photon before scattering

TAPE_ERROR = 0.001																	# 1mm tape accuracy

# ring setup
# L: distance in plane (along symmetry axis), R: real distance to ring (diagonal "in air")
# ========================================================================================

# ring setup parameters (detector area, ring diameters, ring widths, ring volumes, number of electrons in volume)
F_det_ring = np.pi * (ufloat(0.081,TAPE_ERROR,'sys')/2)**2

vD_inner = pl.uarray_tag([0.221, 0.171, 0.121],[TAPE_ERROR]*3,'sys')
vD_outer = pl.uarray_tag([0.250, 0.199, 0.149],[TAPE_ERROR]*3,'sys')
vD = (vD_outer-vD_inner)/2+vD_inner
vD_used = np.append(vD,[vD[-1]]*2)													# rings used: [large,medium,small,small,small]

width = ufloat(0.014,TAPE_ERROR,'sys')
vVolumes = np.pi * (vD_outer**2 - vD_inner**2) * width
# N_e =

# distance ring - detector
vLD_ring_long = pl.uarray_tag([0.23,0.28],[TAPE_ERROR]*2,'sys')
vLD_ring_short = pl.uarray_tag([0.211,0.216],[TAPE_ERROR]*2,'sys')
LD_ring = np.mean( (vLD_ring_long - vLD_ring_short)/2 + vLD_ring_short  )			# take average of both sides as true distance

# find distances ring - source required for given theta angles
vTheta_required = pl.degToSr(np.array([50, 40, 30, 24, 19]))
print("Theta angles needed: \n",pl.srToDeg(vTheta_required))
vLS_ring_required = vD_used / 2 * unp.tan(np.pi - unp.arctan(2 * LD_ring / vD_used) - vTheta_required)
print("Plane distances to source needed: \n", vLS_ring_required)

# now the other way round: get errors on theta angles through set LS distance
# assume distance is set to cm precision of requirement (quite rough)
vLS_ring_set = pl.uarray_tag( np.round(unp.nominal_values(vLS_ring_required),2), [0.01]*vLS_ring_required.size, 'sys')
	# np.array([ufloat(np.round(x.nominal_value, 2), 0.01) for _, x in enumerate(vLS_ring_required)])
print("Plane distances to source set: \n", vLS_ring_set)
vTheta_set = np.pi - unp.arctan(2 * vLS_ring_set / vD_used) - unp.arctan(2 * LD_ring / vD_used)
print("Theta angles set: \n",pl.srToDeg(vTheta_set))

vRD_ring = unp.sqrt(LD_ring ** 2 + (vD_used / 2) ** 2)
vRS_ring = unp.sqrt(vLS_ring_set ** 2 + (vD_used / 2) ** 2)
print("Diagonal distance ring - detector: \n", vRD_ring, "\n\t\t\t\tring - source: \n", vRS_ring)

pl.printAsLatexTable(np.array([['${:.0f}^\circ$'.format(x) for _,x in enumerate(pl.srToDeg(vTheta_required))],
							   ['${:.1ufL}$'.format(x*10**2) for _,x in enumerate(vLS_ring_required)],
							   ['${:.1ufL}$'.format(x*10**2) for _,x in enumerate(vLS_ring_set)],
							   ['${:.1ufL}$'.format(x) for _,x in enumerate(pl.srToDeg(vTheta_set))],
							   ['${:.1ufL}$'.format(x*10**2) for _,x in enumerate(vRD_ring)],
							   ['${:.1ufL}$'.format(x*10**2) for _, x in enumerate(vRS_ring)]]),
					 colTitles=["large","medium","small","small","small"],
					 rowTitles=[r"required $\theta$",r"required $L_S$ (cm)",
							   r"set $L_S$ (cm)",r"set $\theta$",
							   r"$r$ (cm)",r"$r_0$ (cm)"],
					 mathMode=False)

# conventional setup
# ==================================================================================

# detector area, distances scattering body - source/detector
F_det_conv = np.pi * (ufloat(0.026,TAPE_ERROR,'sys')/2)**2
RS_conv = ufloat(0.049,2*TAPE_ERROR,'sys')
RD_conv = ufloat(0.272,2*TAPE_ERROR,'sys')
# r0_conv = 49e-3 # distance source - body in conv.
# dr0_conv = 2e-3
# r_conv = 272e-3 # distance body - detector in conv.
# dr_conv = 2e-3



# d_large = ufloat((0.25 - 0.221) / 2 + 0.221, TAPE_ERROR)
# d_medium = ufloat((0.199 - 0.171) / 2 + 0.171, TAPE_ERROR)
# d_small = ufloat((0.149 - 0.121) / 2 + 0.121, TAPE_ERROR)
# d_tiny = ufloat(0.085, TAPE_ERROR)
# vD_used = np.array([d_large,d_medium,d_small,d_small,d_small])

# distances to detector

# rD_ring_left = ufloat((0.23 - 0.211) / 2 + 0.211, TAPE_ERROR)
# rD_ring_right = ufloat((0.28 - 0.216) / 2 + 0.216, TAPE_ERROR)
# rD_ring = np.mean([rD_ring_left, rD_ring_right])


# # measurements
# Rw = 14e-3 # ring width
# dRw = 1e-3
# Di = np.array([121e-3, 171e-3, 221e-3]) # inner diameter
# dDi = np.full(3, 1e-3)
# Do = np.array([149e-3, 199e-3, 250e-3]) # outer diameter
# dDo = np.full(3, 1e-3)
# Rv = np.pi * (Do**2 - Di**2) * Rw # ring volume
# dRv = 123
# Ne = 123


# longleft = 230e-3 # calc distance body - detector in ring
# dll = 1e-3
# shortleft = 211e-3
# dsl = 1e-3
# longright = 228e-3
# dlr = 1e-3
# shortright = 216e-3
# dsr = 1e-3
# left = (longleft-shortleft)/2 + shortleft
# dleft = 123
# right = (longright-shortright)/2 + shortright
# dright = 123
# r2 = (left+right)/2
# dr2 = 123
# r1 = 273e-3 # distance source - body in ring # whatabout source length?
# dr1 = 1e-3



# cd = 26e-3 # collimator diameter
# rcd = 1e-3
# Fc = np.pi*(cd/2)**2 # detector surface for conv.
# dFc = 123
# F_ring = 123 # detector surface for ring
# dF_ring = 123

# A = 36697224.2834653 # activity on day of experiment
# dA = 123
#
# I = 0.85 # photon yield

def mu_air(E):
	return 123
def mu_al(E):
	return 123

x1_air_ring = 1 # approx
dx1_air_ring = 123
x2_air_ring = 1 # approx
dx2_air_ring = 123
x1_al_ring = 0 # approx
dx1_al_ring = 123
x2_al_ring = 0
dx2_al_ring = 123

x1_air_conv = 1
dx1_air_conv = 123
x2_air_conv = 1
dx2_air_conv = 123
x1_al_conv = 0
dx1_al_conv = 123
x2_al_conv = 0
dx2_al_conv = 123

#def eta_ring(E_prime): # absorbtion
#	return np.exp(-mu_air(E_0)*x1_air_ring)*np.exp(-mu_air(E_0)*x1_al_ring)*np.exp(-mu_air(E_prime)*x2_al_ring)*np.exp(-mu_air(E_prime)*x2_air_ring)
#def eta_conv(E_prime): # absorbtion
#	return np.exp(-mu_air(E_0)*x1_air_conv)*np.exp(-mu_air(E_0)*x1_al_conv)*np.exp(-mu_air(E_prime)*x2_al_conv)*np.exp(-mu_air(E_prime)*x2_air_conv)

# eff = 123

# set strings and angles
method = ['Ring', 'Conv']
material = ['Al', 'Fe']
# angle = [pl.srToDeg(np.flipud(vTheta_set)), pl.uarray_tag([50, 60, 80, 90, 105, 135],[5/np.sqrt(12)]*6,'sys')]
angle = [[19,24,30,40,50],[50, 60, 80, 90, 105, 135]]

# set peak bounds [[Ring], [[Conv_Al], [Conv_Fe]]]
bound = [[[380,480],[375,450],[340,430],[320,385],[280,360]],
		 [[[275,350],[250,310],[200,250],[185,230],[165,200],[145,170]],
		  [[275,350],[250,310],[200,255],[180,240],[160,205],[150,170]]]]

# set lists for gauss fit
mean = [[], [], []]
dmean = [[], [], []]
sig = [[], [], []]
dsig = [[], [], []]
n = [[], [], []]
cross = [[], [], []]
dcross = [[], [], []]

# plot raw data and find peaks
for i, m in enumerate(method):
	if m == 'Conv':
		for j, a in enumerate(angle[i]):
			for k, mat in enumerate(material):
				
				name = str(int(round(a)))+'_'+m+'_'+mat
				data = np.genfromtxt('Data/'+name+'.TKA')
				noise = np.genfromtxt('Data/'+str(int(round(a)))+'_'+m+'_Noise.TKA')
				
				data = data - noise
				data = np.delete(data, [0,1])
				chan = np.arange(len(data))
				
				fig, ax = plt.subplots()
				ax.plot(chan, data, '.')
				ax.set_title(str(int(round(a)))+m+mat)
				fig.savefig('Figures/'+name)
				
				[before, peak, after] = np.split(data, bound[i][k][j])
				[before, seg, after] = np.split(chan, bound[i][k][j])
				
				opt, cov = curve_fit(pl.gauss, seg, peak, p0=[bound[i][k][j][0],1,1])
				mean[i+k] += [opt[0]]
				dmean[i+k] +=  [np.sqrt(cov[0][0])]
				sig[i+k] += [opt[1]]
				dsig[i+k] += [np.sqrt(cov[1][1])]
				n[i+k] += [opt[2]]
				
				# get FWHM
				FWHM = 2 * np.sqrt(2 * np.log(2)) * opt[1]
				dFWHM = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(cov[1][1])
				
				# get counts in peak
				lbound = int(round(opt[0]-FWHM/2))
				rbound = int(round(opt[0]+FWHM/2))
				[before, peak, _] = np.split(data, [lbound,rbound])
				mpeak = np.sum(peak)
				
				temp = mean[i+k]
				dtemp = dmean[i+k]
				#E, dE = ChtoE(temp, dtemp)
				#cross[i+k] = 4*np.pi*r0_conv**2*r_conv**2*mpeak/(eta_conv(E)*eff*Ne_conv*A*I*F_conv)
	
	else:
		for j, a in enumerate(angle[i]):
			
			name = str(int(round(a)))+'_'+m
			data = np.genfromtxt('Data/'+name+'.TKA')
			noise = np.genfromtxt('Data/'+name+'_Noise.TKA')
			
			data = data - noise
			data = np.delete(data, [0,1])
			chan = np.arange(len(data))
			
			fig, ax = plt.subplots()
			ax.plot(chan, data, '.')
			ax.set_title(str(int(round(a)))+m)
			fig.savefig('Figures/'+name)
			
			[before, peak, after] = np.split(data, bound[i][j])
			[before, seg, after] = np.split(chan, bound[i][j])
			
			opt, cov = curve_fit(pl.gauss, seg, peak, p0=[bound[i][j][0],1,1])
			mean[i] += [opt[0]]
			dmean[i] += [np.sqrt(cov[0][0])]
			sig[i] += [opt[1]]
			dsig[i] += [np.sqrt(cov[1][1])]
			n[i] += [opt[2]]

#print(mean)
#plt.plot(chan, data, '.')
#plt.plot(chan, gauss(chan, mean[2][5], sig[2][5], n[2][5]), '-')

# data shape: [5*ring,6*Al,6*Fe]
# angle,dangle,mean,dmean,sig,dsig = np.loadtxt('photo_peaks_2.NORM')
real_angle,real_dangle,_,_,_,_= np.loadtxt('photo_peaks_2.NORM')

# convert channel to energy
mean = np.concatenate(mean)
dmean = np.concatenate(dmean)
E = unp.uarray(mean, dmean)
E = ChtoE(E)

E = unp.nominal_values(E)
dE = unp.std_devs(E)

theta = unp.uarray(real_angle,real_dangle)

# theory
theo1 = 661657*e / (1 + (661657*e/(m_e*c**2))*(1-unp.cos(pl.degToSr(theta[0:11]))))
# theo2 = 661657*e / (1 + (661657*e/(m_e*c**2))*(1-unp.cos(pl.degToSr(np.array(angle[1])))))
theo1 = theo1/(e*1000)
# theo2 = theo2/(e*1000)

fig, ax = plt.subplots()
ax.errorbar(angle[0:5], E[0:5], fmt='b.', xerr=dangle[0:5], yerr=dE[0:5], label='Ring geometry')
ax.errorbar(angle[5:11], E[5:11], fmt='g.', xerr=dangle[5:11], yerr=dE[5:11], label='Conventional geometry, Al cylinder')
ax.errorbar(angle[11:17], E[11:17], fmt='m.', xerr=dangle[11:17], yerr=dE[11:17], label='Conventional geometry, Fe cylinder')
ax.plot(angle[0:5], unp.nominal_values(theo1[0:5]), 'r-', label='Theory prediction with error margin')
ax.plot(angle[0:5], unp.nominal_values(theo1[0:5])+unp.std_devs(theo1[0:5]), 'r--')
ax.plot(angle[0:5], unp.nominal_values(theo1[0:5])-unp.std_devs(theo1[0:5]), 'r--')
ax.plot(angle[5:11], unp.nominal_values(theo1[5:11]), 'r-')
ax.plot(angle[5:11], unp.nominal_values(theo1[5:11])+unp.std_devs(theo1[5:11]), 'r--')
ax.plot(angle[5:11], unp.nominal_values(theo1[5:11])-unp.std_devs(theo1[5:11]), 'r--')

# ax.errorbar(int(round(angle[1])), theo2, 'r-')
ax.legend(loc='upper right')
ax.set_xlabel(r'$\theta [^\circ]$')
ax.set_ylabel(r'$E_{\gamma}^{\prime}$ [keV]')
ax.set_title('Energy of scattered photons vs scattering angle')
fig.savefig('Figures/E_Phi')

# create NORM file
mean = np.concatenate(mean)
dmean = np.concatenate(dmean)
sig = np.concatenate(sig)
dsig = np.concatenate(dsig)
angle = np.append(np.concatenate(angle),angle[1])

np.savetxt('photo_peaks_2.NORM',[unp.nominal_values(angle),unp.std_devs(dangle),mean,dmean,sig,dsig])
