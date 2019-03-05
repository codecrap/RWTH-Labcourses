#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:21:13 2019

@author: alex
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from calibration import ChtoE
from scipy.constants import c, m_e, e 
import sys
sys.path.append("./../../")
import PraktLib as pl

# gauss function
def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

# measurements
Rw = 14e-3 # ring width
dRw = 1e-3
Di = np.array([121e-3, 171e-3, 221e-3]) # inner diameter
dDi = np.full(3, 1e-3)
Do = np.array([149e-3, 199e-3, 250e-3]) # outer diameter
dDo = np.full(3, 1e-3)
Rv = np.pi * (Do**2 - Di**2) * Rw # ring volume
dRv = 123
Ne = 123

r0_conv = 49e-3 # distance source - body in conv.
dr0_conv = 2e-3
r_conv = 272e-3 # distance body - detector in conv.
dr_conv = 2e-3

longleft = 230e-3 # calc distance body - detector in ring
dll = 1e-3
shortleft = 211e-3
dsl = 1e-3
longright = 228e-3
dlr = 1e-3
shortright = 216e-3
dsr = 1e-3
left = (longleft-shortleft)/2 + shortleft
dleft = 123 
right = (longright-shortright)/2 + shortright
dright = 123
r2 = (left+right)/2
dr2 = 123
r1 = 273e-3 # distance source - body in ring # whatabout source length?
dr1 = 1e-3

h = (Do-Di)/4+Di/2
dh = 123
r0_ring = np.sqrt(r1**2 + h**2)
dr0_ring = 123
r_ring = np.sqrt(r2**2 + h**2)
dr_ring = 123

cd = 26e-3 # collimator diameter
rcd = 1e-3
Fc = np.pi*(cd/2)**2 # detector surface for conv. 
dFc = 123
F_ring = 123 # detector surface for ring
dF_ring = 123

A = 36697224.2834653 # activity on day of experiment
dA = 123

I = 0.85 # photon yield

E_0 = 123 # energy of photon before scattering 
def mu_air(E):
    return 123
def mu_al(E):
    return 123

x1_air_ring = r0_ring # approx
dx1_air_ring = 123
x2_air_ring = r_ring # approx
dx2_air_ring = 123
x1_al_ring = 0 # approx
dx1_al_ring = 123
x2_al_ring = 0
dx2_al_ring = 123

x1_air_conv = r0_conv
dx1_air_conv = 123
x2_air_conv = r_conv
dx2_air_conv = 123
x1_al_conv = 0
dx1_al_conv = 123
x2_al_conv = 0
dx2_al_conv = 123

def eta_ring(E_prime): # absorbtion
    return np.exp(-mu_air(E_0)*x1_air_ring)*np.exp(-mu_air(E_0)*x1_al_ring)*np.exp(-mu_air(E_prime)*x2_al_ring)*np.exp(-mu_air(E_prime)*x2_air_ring)
def eta_conv(E_prime): # absorbtion
    return np.exp(-mu_air(E_0)*x1_air_conv)*np.exp(-mu_air(E_0)*x1_al_conv)*np.exp(-mu_air(E_prime)*x2_al_conv)*np.exp(-mu_air(E_prime)*x2_air_conv)

eff = 123

# set strings and angles
method = ['Ring', 'Conv']
material = ['Al', 'Fe']
angle = [[19, 24, 30, 40, 50], [50, 60, 80, 90, 105, 135]]

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
                
                name = str(a)+'_'+m+'_'+mat
                data = np.genfromtxt('Data/'+name+'.TKA')
                noise = np.genfromtxt('Data/'+str(a)+'_'+m+'_Noise.TKA')
                
                data = data - noise
                data = np.delete(data, [0,1])
                chan = np.arange(len(data))
                
                fig, ax = plt.subplots()
                ax.plot(chan, data, '.')
                ax.set_title(name)
                fig.savefig('Figures/'+name+'.pdf',format='pdf',dpi=256)
                
                [before, peak, after] = np.split(data, bound[i][k][j])
                [before, seg, after] = np.split(chan, bound[i][k][j])
                
                opt, cov = curve_fit(gauss, seg, peak, p0=[bound[i][k][j][0],1,1])
                mean[i+k] += [opt[0]]
                dmean[i+k] += [np.sqrt(cov[0][0])]
                sig[i+k] += [opt[1]]
                dsig[i+k] += [np.sqrt(cov[1][1])]
                n[i+k] += [opt[2]]
                
                # get FWHM
                FWHM = 2 * np.sqrt(2 * np.log(2)) * opt[1]
                dFWHM = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(cov[1][1])
                
                # get counts in peak
                lbound = int(round(opt[0]-FWHM/2))
                rbound = int(round(opt[0]+FWHM/2))
                [before, peak, after] = np.split(data, [lbound,rbound])
                mpeak = np.sum(peak)
                
                temp = mean[i+k]
                dtemp = dmean[i+k]
                #E, dE = ChtoE(temp, dtemp)
                #cross[i+k] = 4*np.pi*r0_conv**2*r_conv**2*mpeak/(eta_conv(E)*eff*Ne_conv*A*I*F_conv)
                
    else:
        for j, a in enumerate(angle[i]):
            
            name = str(a)+'_'+m
            data = np.genfromtxt('Data/'+name+'.TKA')
            noise = np.genfromtxt('Data/'+name+'_Noise.TKA')
            
            data = data - noise
            data = np.delete(data, [0,1])
            chan = np.arange(len(data))
            
            fig, ax = plt.subplots()
            ax.plot(chan, data, '.')
            ax.set_title(name)
            fig.savefig('Figures/'+name+'.pdf',format='pdf',dpi=256)
            
            [before, peak, after] = np.split(data, bound[i][j])
            [before, seg, after] = np.split(chan, bound[i][j])
            
            opt, cov = curve_fit(gauss, seg, peak, p0=[bound[i][j][0],1,1])
            mean[i] += [opt[0]]
            dmean[i] += [np.sqrt(cov[0][0])]
            sig[i] += [opt[1]]
            dsig[i] += [np.sqrt(cov[1][1])]
            n[i] += [opt[2]]

#print(mean)
#plt.plot(chan, data, '.')
#plt.plot(chan, gauss(chan, mean[2][5], sig[2][5], n[2][5]), '-')

# convert channel to energy
E = mean
dE = dmean
for i, ival in enumerate(E):
    for j, jval in enumerate(ival):
        en, den = ChtoE(jval, dE[i][j])
        E[i][j] = en
        dE[i][j] = den
        
# theory
theo1 = 661657*e / (1 + (661657*e/(m_e*c**2))*(1-np.cos(pl.degToSr(np.array(angle[0])))))
theo2 = 661657*e / (1 + (661657*e/(m_e*c**2))*(1-np.cos(pl.degToSr(np.array(angle[1])))))
theo1 = theo1/(e*1000)
theo2 = theo2/(e*1000)

fig, ax = plt.subplots()
ax.plot(angle[0], E[0], '.')
ax.plot(angle[1], E[1], '.')
ax.plot(angle[1], E[2], '.')
ax.plot(angle[0], theo1, 'r-')
ax.plot(angle[1], theo2, 'r-')
ax.set_title('energy of scattered photons')
fig.savefig('Figures/E_Phi.pdf',format='pdf',dpi=256)

# create NORM file
mean = np.concatenate(mean)
dmean = np.concatenate(dmean)
sig = np.concatenate(sig)
dsig = np.concatenate(dsig)
temp = angle[1]
angle = np.concatenate(angle)
angle = np.append(angle, temp)

np.savetxt('photo_peaks_2.NORM',[angle,mean,dmean,sig,dsig])
