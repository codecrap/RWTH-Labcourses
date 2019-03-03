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

# set strings
method = ['Ring', 'Conv']
material = ['Al', 'Fe']
angle = [[19, 24, 30, 40, 50], [50, 60, 90]]

# plot raw data
for i, m in enumerate(method):
    if m == 'Conv':
        for j, a in enumerate(angle[i]):
            for k, mat in enumerate(material):
                data = np.genfromtxt('Data/'+str(a)+'_'+m+'_'+mat+'.TKA')
                noise = np.genfromtxt('Data/'+str(a)+'_'+m+'_Noise.TKA')
                data = data - noise
                data = np.delete(data, [0,1])
                chan = np.arange(len(data))
                fig, ax = plt.subplots()
                ax.plot(chan, data, '.')
                fig.savefig('Figures/'+str(a)+'_'+m+'_'+mat+'.pdf',format='pdf',dpi=256)
    else:
        for j, a in enumerate(angle[i]):
            data = np.genfromtxt('Data/'+str(a)+'_'+m+'.TKA')
            noise = np.genfromtxt('Data/'+str(a)+'_'+m+'_Noise.TKA')
            data = data - noise
            data = np.delete(data, [0,1])
            chan = np.arange(len(data))
            fig, ax = plt.subplots()
            ax.plot(chan, data, '.')
            fig.savefig('Figures/'+str(a)+'_'+m+'_'+mat+'.pdf',format='pdf',dpi=256)
