#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:21:13 2019

@author: alex
"""

import numpy as np
#import PraktLib as pl
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# gauss function
def gauss(x, x0, sigma, a):
    return a * np.exp(-(x-x0)**2/(2.*sigma**2))

# set strings
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
