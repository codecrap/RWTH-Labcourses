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

pfad = "Hauptmessung.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_ln = np.log(druck)
temp_T = 1/temperatur

Len = len(druck_ln)

sig_p = 0.348 * np.ones(Len)
sig_T = 0.174 * np.ones(Len)

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
"""
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
"""
''' plot

plt.figure()
plt.plot(temp_T, linreg[0]*temp_T+linreg[2])
plt.errorbar(temp_T, druck_ln, sig_lnp, sig_einsdurchT, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()'''

''' mit slicing '''
pfad = "Hauptmessung.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_1  =  druck[0:1*len(druck)/16+1]
druck_2  =  druck[1*len(druck)/16+1:2*len(druck)/16+1]
druck_3  =  druck[2*len(druck)/16+1:3*len(druck)/16+1]
druck_4  =  druck[3*len(druck)/16+1:4*len(druck)/16+1]
druck_5  =  druck[4*len(druck)/16+1:5*len(druck)/16+1]
druck_6  =  druck[5*len(druck)/16+1:6*len(druck)/16+1]
druck_7  =  druck[6*len(druck)/16+1:7*len(druck)/16+1]
druck_8  =  druck[7*len(druck)/16+1:8*len(druck)/16+1]
druck_9  =  druck[8*len(druck)/16+1:9*len(druck)/16+1]
druck_10  =  druck[9*len(druck)/16+1:10*len(druck)/16+1]
druck_11  =  druck[10*len(druck)/16+1:11*len(druck)/16+1]
druck_12  =  druck[11*len(druck)/16+1:12*len(druck)/16+1]
druck_13  =  druck[12*len(druck)/16+1:13*len(druck)/16+1]
druck_14  =  druck[13*len(druck)/16+1:14*len(druck)/16+1]
druck_15  =  druck[14*len(druck)/16+1:15*len(druck)/16+1]
druck_16  =  druck[15*len(druck)/16+1:16*len(druck)/16+1]

druck_1_ln = np.log(druck[0:1*len(druck)/16+1])
druck_2_ln = np.log(druck[1*len(druck)/16+1:2*len(druck)/16+1])
druck_3_ln = np.log(druck[2*len(druck)/16+1:3*len(druck)/16+1])
druck_4_ln = np.log(druck[3*len(druck)/16+1:4*len(druck)/16+1])
druck_5_ln = np.log(druck[4*len(druck)/16+1:5*len(druck)/16+1])
druck_6_ln = np.log(druck[5*len(druck)/16+1:6*len(druck)/16+1])
druck_7_ln = np.log(druck[6*len(druck)/16+1:7*len(druck)/16+1])
druck_8_ln = np.log(druck[7*len(druck)/16+1:8*len(druck)/16+1])
druck_9_ln = np.log(druck[8*len(druck)/16+1:9*len(druck)/16+1])
druck_10_ln = np.log(druck[9*len(druck)/16+1:10*len(druck)/16+1])
druck_11_ln = np.log(druck[10*len(druck)/16+1:11*len(druck)/16+1])
druck_12_ln = np.log(druck[11*len(druck)/16+1:12*len(druck)/16+1])
druck_13_ln = np.log(druck[12*len(druck)/16+1:13*len(druck)/16+1])
druck_14_ln = np.log(druck[13*len(druck)/16+1:14*len(druck)/16+1])
druck_15_ln = np.log(druck[14*len(druck)/16+1:15*len(druck)/16+1])
druck_16_ln = np.log(druck[15*len(druck)/16+1:16*len(druck)/16+1])

temp_1 =  temperatur[0:1*len(temperatur)/16+1]
temp_2 =  temperatur[1*len(temperatur)/16+1:2*len(temperatur)/16+1]
temp_3 =  temperatur[2*len(temperatur)/16+1:3*len(temperatur)/16+1]
temp_4 =  temperatur[3*len(temperatur)/16+1:4*len(temperatur)/16+1]
temp_2 =  temperatur[4*len(temperatur)/16+1:5*len(temperatur)/16+1]
temp_3 =  temperatur[5*len(temperatur)/16+1:6*len(temperatur)/16+1]
temp_4 =  temperatur[6*len(temperatur)/16+1:7*len(temperatur)/16+1]
temp_2 =  temperatur[7*len(temperatur)/16+1:8*len(temperatur)/16+1]
temp_3 =  temperatur[8*len(temperatur)/16+1:9*len(temperatur)/16+1]
temp_4 =  temperatur[9*len(temperatur)/16+1:10*len(temperatur)/16+1]
temp_2 =  temperatur[10*len(temperatur)/16+1:11*len(temperatur)/16+1]
temp_3 =  temperatur[11*len(temperatur)/16+1:12*len(temperatur)/16+1]
temp_4 =  temperatur[12*len(temperatur)/16+1:13*len(temperatur)/16+1]
temp_2 =  temperatur[13*len(temperatur)/16+1:14*len(temperatur)/16+1]
temp_3 =  temperatur[14*len(temperatur)/16+1:15*len(temperatur)/16+1]
temp_4 =  temperatur[15*len(temperatur)/16+1:16*len(temperatur)/16+1]


temp_1_T = 1/temperatur[0:1*len(temperatur)/16+1]
temp_2_T = 1/temperatur[1*len(temperatur)/16+1:2*len(temperatur)/16+1]
temp_3_T = 1/temperatur[2*len(temperatur)/16+1:3*len(temperatur)/16+1]
temp_4_T = 1/temperatur[3*len(temperatur)/16+1:4*len(temperatur)/16+1]
temp_2_T = 1/temperatur[4*len(temperatur)/16+1:5*len(temperatur)/16+1]
temp_3_T = 1/temperatur[5*len(temperatur)/16+1:6*len(temperatur)/16+1]
temp_4_T = 1/temperatur[6*len(temperatur)/16+1:7*len(temperatur)/16+1]
temp_2_T = 1/temperatur[7*len(temperatur)/16+1:8*len(temperatur)/16+1]
temp_3_T = 1/temperatur[8*len(temperatur)/16+1:9*len(temperatur)/16+1]
temp_4_T = 1/temperatur[9*len(temperatur)/16+1:10*len(temperatur)/16+1]
temp_2_T = 1/temperatur[10*len(temperatur)/16+1:11*len(temperatur)/16+1]
temp_3_T = 1/temperatur[11*len(temperatur)/16+1:12*len(temperatur)/16+1]
temp_4_T = 1/temperatur[12*len(temperatur)/16+1:13*len(temperatur)/16+1]
temp_2_T = 1/temperatur[13*len(temperatur)/16+1:14*len(temperatur)/16+1]
temp_3_T = 1/temperatur[14*len(temperatur)/16+1:15*len(temperatur)/16+1]
temp_4_T = 1/temperatur[15*len(temperatur)/16+1:16*len(temperatur)/16+1]


Len1 = len(druck_1_ln)

sig_p_1 = 0.348 * np.ones(Len1)
sig_T_1 = 0.174 * np.ones(Len1)

sig_lnp_1 = sig_p_1/druck_1
sig_einsdurchT_1 = sig_T_1/(temp_1**2)

sig_lnp_2 = sig_p_1/druck_2
sig_einsdurchT_2 = sig_T_1/(temp_2**2)

sig_lnp_3 = sig_p_1/druck_3
sig_einsdurchT_3 = sig_T_1/(temp_3**2)

sig_lnp_4 = sig_p_1/druck_4
sig_einsdurchT_4 = sig_T_1/(temp_4**2)

sig_lnp_5 = sig_p_1/druck_5
sig_einsdurchT_5 = sig_T_1/(temp_5**2)

sig_lnp_6 = sig_p_1/druck_6
sig_einsdurchT_6 = sig_T_1/(temp_6**2)

sig_lnp_7 = sig_p_1/druck_7
sig_einsdurchT_7 = sig_T_1/(temp_7**2)

sig_lnp_8 = sig_p_1/druck_8
sig_einsdurchT_8 = sig_T_1/(temp_8**2)

sig_lnp_9 = sig_p_1/druck_9
sig_einsdurchT_9 = sig_T_1/(temp_9**2)

sig_lnp_10 = sig_p_1/druck_10
sig_einsdurchT_10 = sig_T_1/(temp_10**2)

sig_lnp_11 = sig_p_1/druck_11
sig_einsdurchT_11 = sig_T_1/(temp_11**2)

sig_lnp_12 = sig_p_1/druck_12
sig_einsdurchT_12 = sig_T_1/(temp_12**2)

sig_lnp_13 = sig_p_1/druck_13
sig_einsdurchT_13 = sig_T_1/(temp_13**2)

sig_lnp_14 = sig_p_1/druck_14
sig_einsdurchT_14 = sig_T_1/(temp_14**2)

sig_lnp_15 = sig_p_1/druck_15
sig_einsdurchT_15 = sig_T_1/(temp_15**2)

sig_lnp_16 = sig_p_1/druck_16
sig_einsdurchT_16 = sig_T_1/(temp_16**2)

linreg_1 = p.lineare_regression_xy(temp_1_T, druck_1_ln, sig_einsdurchT_1, sig_lnp_1)

linreg_2 = p.lineare_regression_xy(temp_2_T, druck_2_ln, sig_einsdurchT_2, sig_lnp_2)

linreg_3 = p.lineare_regression_xy(temp_3_T, druck_3_ln, sig_einsdurchT_3, sig_lnp_3)

linreg_4 = p.lineare_regression_xy(temp_4_T, druck_4_ln, sig_einsdurchT_4, sig_lnp_4)

linreg_5 = p.lineare_regression_xy(temp_5_T, druck_5_ln, sig_einsdurchT_5, sig_lnp_5)

linreg_6 = p.lineare_regression_xy(temp_6_T, druck_6_ln, sig_einsdurchT_6, sig_lnp_6)

linreg_7 = p.lineare_regression_xy(temp_7_T, druck_7_ln, sig_einsdurchT_7, sig_lnp_7)

linreg_8 = p.lineare_regression_xy(temp_8_T, druck_8_ln, sig_einsdurchT_8, sig_lnp_8)

linreg_9 = p.lineare_regression_xy(temp_9_T, druck_9_ln, sig_einsdurchT_9, sig_lnp_9)

linreg_10 = p.lineare_regression_xy(temp_10_T, druck_10_ln, sig_einsdurchT_10, sig_lnp_10)

linreg_11 = p.lineare_regression_xy(temp_11_T, druck_11_ln, sig_einsdurchT_11, sig_lnp_11)

linreg_12 = p.lineare_regression_xy(temp_12_T, druck_12_ln, sig_einsdurchT_12, sig_lnp_12)

linreg_13 = p.lineare_regression_xy(temp_13_T, druck_13_ln, sig_einsdurchT_13, sig_lnp_13)

linreg_14 = p.lineare_regression_xy(temp_14_T, druck_14_ln, sig_einsdurchT_14, sig_lnp_14)

linreg_15 = p.lineare_regression_xy(temp_15_T, druck_15_ln, sig_einsdurchT_15, sig_lnp_15)

linreg_16 = p.lineare_regression_xy(temp_16_T, druck_16_ln, sig_einsdurchT_16, sig_lnp_16)

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
''' linreg_4 raus!!!'''
plt.figure()
plt.plot(temp_4_T, linreg_4[0]*temp_4_T+linreg_4[2])
plt.errorbar(temp_4_T, druck_4_ln, sig_lnp_4, sig_einsdurchT_4, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
