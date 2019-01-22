# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:43:28 2016

@author: Defender833
"""

import Praktikum as p
import einlesen
import numpy as np
import matplotlib.pyplot as plt

''' Gruppe 1 '''

''' einlesen der Daten '''
pfad = "dichtigkeit.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

k = 4000

druck_k = druck[len(druck)-k:len(druck)+1]
zeit_k = zeit[len(zeit)-k:len(zeit)+1]

sig_p = 0.370 * np.ones(len(druck_k))
sig_t = 0.050 / np.sqrt(12) * np.ones(len(zeit_k))

linreg = p.lineare_regression_xy(zeit_k, druck_k, sig_t, sig_p)
print linreg[4]/(len(druck_k)-2), "Chi^2/f"
print linreg[0]*60, "hPa/min"

''' plot 
plt.figure()
plt.plot(zeit_k, linreg[0]*zeit_k+linreg[2])
plt.errorbar(zeit_k, druck_k, sig_p, sig_t, fmt=".")
plt.xlabel('t in s')
plt.ylabel('p in hPa')
plt.show()
'''
''' residuen '''
'''
residuum = druck_k-(linreg[0]*zeit_k+linreg[2])
sig_res = np.sqrt(sig_p**2 + (zeit_k * linreg[1])**2 + linreg[3]**2)

plt.figure()
plt.errorbar(zeit_k, residuum, yerr=sig_res, fmt=".")
#plt.hlines()
plt.xlabel('t in s')
plt.ylabel('Residuen in hPa')
plt.show()
'''

''' Gruppe 2 '''

''' einlesen der Daten '''
pfad = "leak_5.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

k = 4000

#druck_k = druck[len(druck)-k:len(druck)+1]
#zeit_k = zeit[len(zeit)-k:len(zeit)+1]

sig_p = 0.348 * np.ones(len(druck))
sig_t = 0.020 / np.sqrt(12) * np.ones(len(zeit))

linreg = p.lineare_regression_xy(zeit, druck, sig_t, sig_p)
print linreg[4]/(len(druck)-2), "Chi^2/f"
print linreg[0]*60, "hPa/min"

''' plot '''
'''
plt.figure()
plt.plot(zeit, linreg[0]*zeit+linreg[2])
plt.errorbar(zeit, druck, sig_p, sig_t, fmt=".")
plt.xlabel('t in s')
plt.ylabel('p in hPa')
plt.show()
'''

''' residuen '''
'''
residuum = druck-(linreg[0]*zeit+linreg[2])
sig_res = np.sqrt(sig_p**2 + (zeit * linreg[1])**2 + linreg[3]**2)

plt.figure()
plt.errorbar(zeit, residuum, yerr=sig_res, fmt=".")
#plt.hlines()
plt.xlabel('t in s')
plt.ylabel('Residuen in hPa')
plt.show()
'''