# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:55:57 2016

@author: Defender833
"""

import Praktikum as p
import einlesen
import numpy as np
import matplotlib.pyplot as plt

''' Gruppe 1 '''

pfad = "kuhlung_1.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_ln = np.log(druck)
temp_T = 1/temperatur

Len = len(druck_ln)

sig_p = 0.370 * np.ones(Len)
sig_T = 0.054 * np.ones(Len)

sig_lnp = sig_p/druck
sig_einsdurchT = sig_T/(temperatur**2)

linreg = p.lineare_regression_xy(temp_T, druck_ln, sig_einsdurchT, sig_lnp)
print linreg
print linreg[4]/(Len-2)

''' plot 

plt.figure()
plt.plot(temp_T, linreg[0]*temp_T+linreg[2])
plt.errorbar(temp_T, druck_ln, sig_lnp, sig_einsdurchT, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()

'''

''' Messung 2 '''
pfad = "kuhlung_2.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_ln = np.log(druck)
temp_T = 1/temperatur

Len = len(druck_ln)

sig_p = 0.370 * np.ones(Len)
sig_T = 0.054 * np.ones(Len)

sig_lnp = sig_p/druck
sig_einsdurchT = sig_T/(temperatur**2)

linreg = p.lineare_regression_xy(temp_T, druck_ln, sig_einsdurchT, sig_lnp)
print linreg
print linreg[4]/(Len-2)

''' plot

plt.figure()
plt.plot(temp_T, linreg[0]*temp_T+linreg[2])
plt.errorbar(temp_T, druck_ln, sig_lnp, sig_einsdurchT, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()'''

''' mit slicing '''
pfad = "kuhlung_1.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_1 = druck[0:1*len(druck)/4+1]
druck_2 = druck[1*len(druck)/4+1:2*len(druck)/4+1]
druck_3 = druck[2*len(druck)/4+1:3*len(druck)/4+1]
druck_4 = druck[3*len(druck)/4+1:4*len(druck)/4+1]

druck_1_ln = np.log(druck_1)
druck_2_ln = np.log(druck_2)
druck_3_ln = np.log(druck_3)
druck_4_ln = np.log(druck_4)

temp_1 = temperatur[0:1*len(temperatur)/4+1]
temp_2 = temperatur[1*len(temperatur)/4+1:2*len(temperatur)/4+1]
temp_3 = temperatur[2*len(temperatur)/4+1:3*len(temperatur)/4+1]
temp_4 = temperatur[3*len(temperatur)/4+1:4*len(temperatur)/4+1]

temp_1_T = 1/temp_1
temp_2_T = 1/temp_2
temp_3_T = 1/temp_3
temp_4_T = 1/temp_4

Len1 = len(druck_1)
Len2 = len(druck_2)
Len3 = len(druck_3)
Len4 = len(druck_4)

sig_p_1 = 0.370 * np.ones(Len1)
sig_T_1 = 0.054 * np.ones(Len1)

sig_p_2 = 0.370 * np.ones(Len2)
sig_T_2 = 0.054 * np.ones(Len2)

sig_p_3 = 0.370 * np.ones(Len3)
sig_T_3 = 0.054 * np.ones(Len3)

sig_p_4 = 0.370 * np.ones(Len4)
sig_T_4 = 0.054 * np.ones(Len4)

sig_lnp_1 = sig_p_1/druck_1
sig_einsdurchT_1 = sig_T_1/(temp_1**2)

sig_lnp_2 = sig_p_2/druck_2
sig_einsdurchT_2 = sig_T_2/(temp_2**2)

sig_lnp_3 = sig_p_3/druck_3
sig_einsdurchT_3 = sig_T_3/(temp_3**2)

sig_lnp_4 = sig_p_4/druck_4
sig_einsdurchT_4 = sig_T_4/(temp_4**2)

linreg_1 = p.lineare_regression_xy(temp_1_T, druck_1_ln, sig_einsdurchT_1, sig_lnp_1)

linreg_2 = p.lineare_regression_xy(temp_2_T, druck_2_ln, sig_einsdurchT_2, sig_lnp_2)

linreg_3 = p.lineare_regression_xy(temp_3_T, druck_3_ln, sig_einsdurchT_3, sig_lnp_3)

linreg_4 = p.lineare_regression_xy(temp_4_T, druck_4_ln, sig_einsdurchT_4, sig_lnp_4)

print linreg_1
print linreg_1[4]/(Len1-2)
print linreg_2
print linreg_2[4]/(Len2-2)
print linreg_3
print linreg_3[4]/(Len3-2)
print linreg_4
print linreg_4[4]/(Len4-2)

''' plots '''
#%%
''' linreg_1 '''
plt.figure()
plt.plot(temp_1_T, linreg_1[0]*temp_1_T+linreg_1[2])
plt.errorbar(temp_1_T, druck_1_ln, sig_lnp_1, sig_einsdurchT_1, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
''' linreg_2 '''
plt.figure()
plt.plot(temp_2_T, linreg_2[0]*temp_2_T+linreg_2[2])
plt.errorbar(temp_2_T, druck_2_ln, sig_lnp_2, sig_einsdurchT_2, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
''' linreg_3 '''
plt.figure()
plt.plot(temp_3_T, linreg_3[0]*temp_3_T+linreg_3[2])
plt.errorbar(temp_3_T, druck_3_ln, sig_lnp_3, sig_einsdurchT_3, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
''' linreg_4 '''
plt.figure()
plt.plot(temp_4_T, linreg_4[0]*temp_4_T+linreg_4[2])
plt.errorbar(temp_4_T, druck_4_ln, sig_lnp_4, sig_einsdurchT_4, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
