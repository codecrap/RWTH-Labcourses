# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:59 2016

@author: Defender833
"""

import numpy as np
import Praktikum as p
import matplotlib.pyplot as plt

''' stehende Welle bei 2.4k Hz'''

Peaks = np.zeros(4)
Peaks[0] = 1.5
Peaks[1] = 2.5
Peaks[2] = 3.5
Peaks[3] = 4.5

laenge = np.zeros(4)
laenge[0] = (0.025 + 1 * 0.005) - 0.425
laenge[1] = (0.025 + 15 * 0.005) - 0.425
laenge[2] = (0.025 + 30 * 0.005) - 0.425
laenge[3] = (0.025 + 45 * 0.005) - 0.425
#print laenge
sig_l = np.ones(4)*0.001 * 2 * np.sqrt(2)

lin_reg = p.lineare_regression(Peaks, laenge, sig_l)

sig_v = np.sqrt((2400**2) * (lin_reg[1]**2)*2 + (lin_reg[0]*2)**2 * (10**2))

print lin_reg[0], " +- ", sig_v
#print lin_reg[4]/2
#print lin_reg[1]*2
'''
x = np.linspace(0,5,num="1000")
plt.xlabel("Baeuche", fontsize='large')
plt.ylabel("Laenge die das Mikro in das Rohr verschoben wurde", fontsize='large')
plt.errorbar(Peaks, laenge, yerr = sig_l,fmt=".")
plt.plot(x, lin_reg[0]*x + lin_reg[2], label='Lin_Reg')
plt.legend()
'''
''' Residuen '''

residuum = (laenge)-(lin_reg[0]*Peaks + lin_reg[2])
#print residuum

plt.ylabel("Residuen", fontsize='large')
plt.xlabel("Baeuche", fontsize='large')
plt.errorbar(Peaks, residuum, yerr = sig_l, fmt=".")
plt.hlines(0, 0, 5)
plt.legend()
