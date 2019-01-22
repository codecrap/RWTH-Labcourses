# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:55:57 2016

@author: Defender833
"""

import Praktikum as p
import einlesen
import numpy as np
import matplotlib.pyplot as plt

''' Gruppe 2 '''

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
#print linreg
#print linreg[4]/(Len-2)

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
pfad = "Kuhlung_1.lab"
data = einlesen.PraktLib(pfad, 'cassy').getdata()

messpunkt = data[:, 0]
zeit = data[:, 1]
druck = data[:, 2]
temperatur = data[:, 3]

druck_1   =  druck[0:1*len(druck)/16+1]
druck_2   =  druck[1*len(druck)/16+1:2*len(druck)/16+1]
druck_3   =  druck[2*len(druck)/16+1:3*len(druck)/16+1]
druck_4   =  druck[3*len(druck)/16+1:4*len(druck)/16+1]
druck_5   =  druck[4*len(druck)/16+1:5*len(druck)/16+1]
druck_6   =  druck[5*len(druck)/16+1:6*len(druck)/16+1]
druck_7   =  druck[6*len(druck)/16+1:7*len(druck)/16+1]
druck_8   =  druck[7*len(druck)/16+1:8*len(druck)/16+1]
druck_9   =  druck[8*len(druck)/16+1:9*len(druck)/16+1]
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
temp_5 =  temperatur[4*len(temperatur)/16+1:5*len(temperatur)/16+1]
temp_6 =  temperatur[5*len(temperatur)/16+1:6*len(temperatur)/16+1]
temp_7 =  temperatur[6*len(temperatur)/16+1:7*len(temperatur)/16+1]
temp_8 =  temperatur[7*len(temperatur)/16+1:8*len(temperatur)/16+1]
temp_9 =  temperatur[8*len(temperatur)/16+1:9*len(temperatur)/16+1]
temp_10 =  temperatur[9*len(temperatur)/16+1:10*len(temperatur)/16+1]
temp_11 =  temperatur[10*len(temperatur)/16+1:11*len(temperatur)/16+1]
temp_12 =  temperatur[11*len(temperatur)/16+1:12*len(temperatur)/16+1]
temp_13 =  temperatur[12*len(temperatur)/16+1:13*len(temperatur)/16+1]
temp_14 =  temperatur[13*len(temperatur)/16+1:14*len(temperatur)/16+1]
temp_15 =  temperatur[14*len(temperatur)/16+1:15*len(temperatur)/16+1]
temp_16 =  temperatur[15*len(temperatur)/16+1:16*len(temperatur)/16+1]


temp_1_T = 1/temperatur[0:1*len(temperatur)/16+1]
temp_2_T = 1/temperatur[1*len(temperatur)/16+1:2*len(temperatur)/16+1]
temp_3_T = 1/temperatur[2*len(temperatur)/16+1:3*len(temperatur)/16+1]
temp_4_T = 1/temperatur[3*len(temperatur)/16+1:4*len(temperatur)/16+1]
temp_5_T = 1/temperatur[4*len(temperatur)/16+1:5*len(temperatur)/16+1]
temp_6_T = 1/temperatur[5*len(temperatur)/16+1:6*len(temperatur)/16+1]
temp_7_T = 1/temperatur[6*len(temperatur)/16+1:7*len(temperatur)/16+1]
temp_8_T = 1/temperatur[7*len(temperatur)/16+1:8*len(temperatur)/16+1]
temp_9_T = 1/temperatur[8*len(temperatur)/16+1:9*len(temperatur)/16+1]
temp_10_T = 1/temperatur[9*len(temperatur)/16+1:10*len(temperatur)/16+1]
temp_11_T = 1/temperatur[10*len(temperatur)/16+1:11*len(temperatur)/16+1]
temp_12_T = 1/temperatur[11*len(temperatur)/16+1:12*len(temperatur)/16+1]
temp_13_T = 1/temperatur[12*len(temperatur)/16+1:13*len(temperatur)/16+1]
temp_14_T = 1/temperatur[13*len(temperatur)/16+1:14*len(temperatur)/16+1]
temp_15_T = 1/temperatur[14*len(temperatur)/16+1:15*len(temperatur)/16+1]
temp_16_T = 1/temperatur[15*len(temperatur)/16+1:16*len(temperatur)/16+1]


Len1 = len(druck_1)
Len2 = len(druck_2)
Len3 = len(druck_3)
Len4 = len(druck_4)
Len5 = len(druck_5)
Len6 = len(druck_6)
Len7 = len(druck_7)
Len8 = len(druck_8)
Len9 = len(druck_9)
Len10 = len(druck_10)
Len11 = len(druck_11)
Len12 = len(druck_12)
Len13 = len(druck_13)
Len14 = len(druck_14)
Len15 = len(druck_15)
Len16 = len(druck_16)

sig_p_b = 0.370
sig_T_b = 0.054

#sig_p_b = 0.347
#sig_T_b = 0.069

sig_p_1 = sig_p_b * np.ones(Len1)
sig_T_1 = sig_T_b * np.ones(Len1)

sig_p_2 = sig_p_b * np.ones(Len2)
sig_T_2 = sig_T_b * np.ones(Len2)

sig_p_3 = sig_p_b * np.ones(Len3)
sig_T_3 = sig_T_b * np.ones(Len3)

sig_p_4 = sig_p_b * np.ones(Len4)
sig_T_4 = sig_T_b * np.ones(Len4)

sig_p_5 = sig_p_b * np.ones(Len5)
sig_T_5 = sig_T_b * np.ones(Len5)

sig_p_6 = sig_p_b * np.ones(Len6)
sig_T_6 = sig_T_b * np.ones(Len6)

sig_p_7 = sig_p_b * np.ones(Len7)
sig_T_7 = sig_T_b * np.ones(Len7)

sig_p_8 = sig_p_b * np.ones(Len8)
sig_T_8 = sig_T_b * np.ones(Len8)

sig_p_9 = sig_p_b * np.ones(Len9)
sig_T_9 = sig_T_b * np.ones(Len9)

sig_p_10 = sig_p_b * np.ones(Len10)
sig_T_10 = sig_T_b * np.ones(Len10)

sig_p_11 = sig_p_b * np.ones(Len11)
sig_T_11 = sig_T_b * np.ones(Len11)

sig_p_12 = sig_p_b * np.ones(Len12)
sig_T_12 = sig_T_b * np.ones(Len12)

sig_p_13 = sig_p_b * np.ones(Len13)
sig_T_13 = sig_T_b * np.ones(Len13)

sig_p_14 = sig_p_b * np.ones(Len14)
sig_T_14 = sig_T_b * np.ones(Len14)

sig_p_15 = sig_p_b * np.ones(Len15)
sig_T_15 = sig_T_b * np.ones(Len15)

sig_p_16 = sig_p_b * np.ones(Len16)
sig_T_16 = sig_T_b * np.ones(Len16)

sig_lnp_1 = sig_p_1/druck_1
sig_einsdurchT_1 = sig_T_1/(temp_1**2)

sig_lnp_2 = sig_p_2/druck_2
sig_einsdurchT_2 = sig_T_2/(temp_2**2)

sig_lnp_3 = sig_p_3/druck_3
sig_einsdurchT_3 = sig_T_3/(temp_3**2)

sig_lnp_4 = sig_p_4/druck_4
sig_einsdurchT_4 = sig_T_4/(temp_4**2)

sig_lnp_5 = sig_p_5/druck_5
sig_einsdurchT_5 = sig_T_5/(temp_5**2)

sig_lnp_6 = sig_p_6/druck_6
sig_einsdurchT_6 = sig_T_6/(temp_6**2)

sig_lnp_7 = sig_p_7/druck_7
sig_einsdurchT_7 = sig_T_7/(temp_7**2)

sig_lnp_8 = sig_p_8/druck_8
sig_einsdurchT_8 = sig_T_8/(temp_8**2)

sig_lnp_9 = sig_p_9/druck_9
sig_einsdurchT_9 = sig_T_9/(temp_9**2)

sig_lnp_10 = sig_p_10/druck_10
sig_einsdurchT_10 = sig_T_10/(temp_10**2)

sig_lnp_11 = sig_p_11/druck_11
sig_einsdurchT_11 = sig_T_11/(temp_11**2)

sig_lnp_12 = sig_p_12/druck_12
sig_einsdurchT_12 = sig_T_12/(temp_12**2)

sig_lnp_13 = sig_p_13/druck_13
sig_einsdurchT_13 = sig_T_13/(temp_13**2)

sig_lnp_14 = sig_p_14/druck_14
sig_einsdurchT_14 = sig_T_14/(temp_14**2)

sig_lnp_15 = sig_p_15/druck_15
sig_einsdurchT_15 = sig_T_15/(temp_15**2)

sig_lnp_16 = sig_p_16/druck_16
sig_einsdurchT_16 = sig_T_16/(temp_16**2)

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


#print linreg_1[0]
#print linreg_1[4]/(Len1-2)
'''
print linreg_2[0]
print linreg_2[4]/(Len2-2)
print linreg_3[0]
print linreg_3[4]/(Len3-2)
print linreg_4[0]
print linreg_4[4]/(Len4-2)
print linreg_5[0]
print linreg_5[4]/(Len5-2)
print linreg_6[0]
print linreg_6[4]/(Len6-2)
print linreg_7[0]
print linreg_7[4]/(Len7-2)
print linreg_8[0]
print linreg_8[4]/(Len8-2)
print linreg_9[0]
print linreg_9[4]/(Len9-2)
print linreg_10[0]
print linreg_10[4]/(Len10-2)
print linreg_11[0]
print linreg_11[4]/(Len11-2)
print linreg_12[0]
print linreg_12[4]/(Len12-2)
print linreg_13[0]
print linreg_13[4]/(Len13-2)
print linreg_14[0]
print linreg_14[4]/(Len14-2)
print linreg_15[0]
print linreg_15[4]/(Len15-2)
print linreg_16[0]
print linreg_16[4]/(Len16-2)
'''

R = 8.314472
La = np.zeros(16)

La[0] = linreg_1[0]
La[1] = linreg_2[0]
La[2] = linreg_3[0]
La[3] = linreg_4[0]
La[4] = linreg_5[0]
La[5] = linreg_6[0]
La[6] = linreg_7[0]
La[7] = linreg_8[0]
La[8] = linreg_9[0]
La[9] = linreg_10[0]
La[10] = linreg_11[0]
La[11] = linreg_12[0]
La[12] = linreg_13[0]
La[13] = linreg_14[0]
La[14] = linreg_15[0]
La[15] = linreg_16[0]

La *= R / 1000 * (-1)
#print La

sig_la = np.zeros(16)

sig_la[0] = linreg_1[1]
sig_la[1] = linreg_2[1]
sig_la[2] = linreg_3[1]
sig_la[3] = linreg_4[1]
sig_la[4] = linreg_5[1]
sig_la[5] = linreg_6[1]
sig_la[6] = linreg_7[1]
sig_la[7] = linreg_8[1]
sig_la[8] = linreg_9[1]
sig_la[9] = linreg_10[1]
sig_la[10] = linreg_11[1]
sig_la[11] = linreg_12[1]
sig_la[12] = linreg_13[1]
sig_la[13] = linreg_14[1]
sig_la[14] = linreg_15[1]
sig_la[15] = linreg_16[1]

sig_la *= R / 1000
#print sig_la

temp = np.zeros(16)
temp[0] = temp_1.mean()
temp[1] = temp_2.mean()
temp[2] = temp_3.mean()
temp[3] = temp_4.mean()
temp[4] = temp_5.mean()
temp[5] = temp_6.mean()
temp[6] = temp_7.mean()
temp[7] = temp_8.mean()
temp[8] = temp_9.mean()
temp[9] = temp_10.mean()
temp[10] = temp_11.mean()
temp[11] = temp_12.mean()
temp[12] = temp_13.mean()
temp[13] = temp_14.mean()
temp[14] = temp_15.mean()
temp[15] = temp_16.mean()

#print temp
'''
plt.figure()
plt.plot(temp, La)
plt.errorbar(temp, La, yerr=sig_la, fmt=".")
plt.xlabel('T in k')
plt.ylabel('La in kJ/mol')
plt.show()
'''
''' plots '''
#%%
''' linreg_1 '''
'''
plt.figure()
plt.plot(temp_1_T, linreg_1[0]*temp_1_T+linreg_1[2])
plt.errorbar(temp_1_T, druck_1_ln, sig_lnp_1, sig_einsdurchT_1, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()

#%%
'''''' linreg_2 '''
'''plt.figure()
plt.plot(temp_2_T, linreg_2[0]*temp_2_T+linreg_2[2])
plt.errorbar(temp_2_T, druck_2_ln, sig_lnp_2, sig_einsdurchT_2, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
'''
#%%
"""
''' linreg_3 '''
plt.figure()
plt.plot(temp_3_T, linreg_3[0]*temp_3_T+linreg_3[2])
plt.errorbar(temp_3_T, druck_3_ln, sig_lnp_3, sig_einsdurchT_3, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
"""
#%%
''' linreg_4 '''
'''plt.figure()
plt.plot(temp_4_T, linreg_4[0]*temp_4_T+linreg_4[2])
plt.errorbar(temp_4_T, druck_4_ln, sig_lnp_4, sig_einsdurchT_4, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_5 '''
'''plt.figure()
plt.plot(temp_5_T, linreg_5[0]*temp_5_T+linreg_5[2])
plt.errorbar(temp_5_T, druck_5_ln, sig_lnp_5, sig_einsdurchT_5, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_6 '''
'''plt.figure()
plt.plot(temp_6_T, linreg_6[0]*temp_6_T+linreg_6[2])
plt.errorbar(temp_6_T, druck_6_ln, sig_lnp_6, sig_einsdurchT_6, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_7 '''
'''plt.figure()
plt.plot(temp_7_T, linreg_7[0]*temp_7_T+linreg_7[2])
plt.errorbar(temp_7_T, druck_7_ln, sig_lnp_7, sig_einsdurchT_7, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_8 '''
'''plt.figure()
plt.plot(temp_8_T, linreg_8[0]*temp_8_T+linreg_8[2])
plt.errorbar(temp_8_T, druck_8_ln, sig_lnp_8, sig_einsdurchT_8, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_9 '''
'''plt.figure()
plt.plot(temp_9_T, linreg_9[0]*temp_9_T+linreg_9[2])
plt.errorbar(temp_9_T, druck_9_ln, sig_lnp_9, sig_einsdurchT_9, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_10 '''
'''plt.figure()
plt.plot(temp_10_T, linreg_10[0]*temp_10_T+linreg_10[2])
plt.errorbar(temp_10_T, druck_10_ln, sig_lnp_10, sig_einsdurchT_10, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_11 '''
'''plt.figure()
plt.plot(temp_11_T, linreg_11[0]*temp_11_T+linreg_11[2])
plt.errorbar(temp_11_T, druck_11_ln, sig_lnp_11, sig_einsdurchT_11, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_12 '''
'''plt.figure()
plt.plot(temp_12_T, linreg_12[0]*temp_12_T+linreg_12[2])
plt.errorbar(temp_12_T, druck_12_ln, sig_lnp_12, sig_einsdurchT_12, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_13 '''
'''plt.figure()
plt.plot(temp_13_T, linreg_13[0]*temp_13_T+linreg_13[2])
plt.errorbar(temp_13_T, druck_13_ln, sig_lnp_13, sig_einsdurchT_13, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_14 '''
'''plt.figure()
plt.plot(temp_14_T, linreg_14[0]*temp_14_T+linreg_14[2])
plt.errorbar(temp_14_T, druck_14_ln, sig_lnp_14, sig_einsdurchT_14, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_15 '''
'''plt.figure()
plt.plot(temp_15_T, linreg_15[0]*temp_15_T+linreg_15[2])
plt.errorbar(temp_15_T, druck_15_ln, sig_lnp_15, sig_einsdurchT_15, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
#%%
'''''' linreg_16 '''
'''plt.figure()
plt.plot(temp_16_T, linreg_16[0]*temp_16_T+linreg_16[2])
plt.errorbar(temp_16_T, druck_16_ln, sig_lnp_16, sig_einsdurchT_16, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()'''
#%%
'''
plt.figure()
plt.plot(temp_1_T, linreg_1[0]*temp_1_T+linreg_1[2])
#plt.errorbar(temp_1_T, druck_1_ln, sig_lnp_1, sig_einsdurchT_1, fmt=".")
plt.plot(temp_2_T, linreg_2[0]*temp_2_T+linreg_2[2])
#plt.errorbar(temp_2_T, druck_2_ln, sig_lnp_2, sig_einsdurchT_2, fmt=".")
plt.plot(temp_3_T, linreg_3[0]*temp_3_T+linreg_3[2])
#plt.errorbar(temp_3_T, druck_3_ln, sig_lnp_3, sig_einsdurchT_3, fmt=".")
plt.plot(temp_4_T, linreg_4[0]*temp_4_T+linreg_4[2])
#plt.errorbar(temp_4_T, druck_4_ln, sig_lnp_4, sig_einsdurchT_4, fmt=".")
plt.plot(temp_5_T, linreg_5[0]*temp_5_T+linreg_5[2])
#plt.errorbar(temp_5_T, druck_5_ln, sig_lnp_5, sig_einsdurchT_5, fmt=".")
plt.plot(temp_6_T, linreg_6[0]*temp_6_T+linreg_6[2])
#plt.errorbar(temp_6_T, druck_6_ln, sig_lnp_6, sig_einsdurchT_6, fmt=".")
plt.plot(temp_7_T, linreg_7[0]*temp_7_T+linreg_7[2])
#plt.errorbar(temp_7_T, druck_7_ln, sig_lnp_7, sig_einsdurchT_7, fmt=".")
plt.plot(temp_8_T, linreg_8[0]*temp_8_T+linreg_8[2])
#plt.errorbar(temp_8_T, druck_8_ln, sig_lnp_8, sig_einsdurchT_8, fmt=".")
plt.plot(temp_9_T, linreg_9[0]*temp_9_T+linreg_9[2])
#plt.errorbar(temp_9_T, druck_9_ln, sig_lnp_9, sig_einsdurchT_9, fmt=".")
plt.plot(temp_10_T, linreg_10[0]*temp_10_T+linreg_10[2])
#plt.errorbar(temp_10_T, druck_10_ln, sig_lnp_10, sig_einsdurchT_10, fmt=".")
plt.plot(temp_11_T, linreg_11[0]*temp_11_T+linreg_11[2])
#plt.errorbar(temp_11_T, druck_11_ln, sig_lnp_11, sig_einsdurchT_11, fmt=".")
plt.plot(temp_12_T, linreg_12[0]*temp_12_T+linreg_12[2])
#plt.errorbar(temp_12_T, druck_12_ln, sig_lnp_12, sig_einsdurchT_12, fmt=".")
plt.plot(temp_13_T, linreg_13[0]*temp_13_T+linreg_13[2])
#plt.errorbar(temp_13_T, druck_13_ln, sig_lnp_13, sig_einsdurchT_13, fmt=".")
plt.plot(temp_14_T, linreg_14[0]*temp_14_T+linreg_14[2])
#plt.errorbar(temp_14_T, druck_14_ln, sig_lnp_14, sig_einsdurchT_14, fmt=".")
plt.plot(temp_15_T, linreg_15[0]*temp_15_T+linreg_15[2])
#plt.errorbar(temp_15_T, druck_15_ln, sig_lnp_15, sig_einsdurchT_15, fmt=".")
plt.plot(temp_16_T, linreg_16[0]*temp_16_T+linreg_16[2])
#plt.errorbar(temp_16_T, druck_16_ln, sig_lnp_16, sig_einsdurchT_16, fmt=".")
plt.xlabel('1/T in k')
plt.ylabel('ln(p) in hPa')
plt.show()
'''
#%%

''' residuen '''

residuum_0 = np.zeros(Len1)
residuum_1 = np.zeros(Len2)
residuum_2 = np.zeros(Len3)
residuum_3 = np.zeros(Len4)
residuum_4 = np.zeros(Len5)
residuum_5 = np.zeros(Len6)
residuum_6 = np.zeros(Len7)
residuum_7 = np.zeros(Len8)
residuum_8 = np.zeros(Len9)
residuum_9 = np.zeros(Len10)
residuum_10 = np.zeros(Len11)
residuum_11 = np.zeros(Len12)
residuum_12 = np.zeros(Len13)
residuum_13 = np.zeros(Len14)
residuum_14 = np.zeros(Len15)
residuum_15 = np.zeros(Len16)

residuum_0 = druck_1_ln - (linreg_1[0] * temp_1_T + linreg_1[2])
residuum_1 = druck_2_ln - (linreg_2[0] * temp_2_T + linreg_2[2])
residuum_2 = druck_3_ln - (linreg_3[0] * temp_3_T + linreg_3[2])
residuum_3 = druck_4_ln - (linreg_4[0] * temp_4_T + linreg_4[2])
residuum_4 = druck_5_ln - (linreg_5[0] * temp_5_T + linreg_5[2])
residuum_5 = druck_6_ln - (linreg_6[0] * temp_6_T + linreg_6[2])
residuum_6 = druck_7_ln - (linreg_7[0] * temp_7_T + linreg_7[2])
residuum_7 = druck_8_ln - (linreg_8[0] * temp_8_T + linreg_8[2])
residuum_8 = druck_9_ln - (linreg_9[0] * temp_9_T + linreg_9[2])
residuum_9 = druck_10_ln - (linreg_10[0] * temp_10_T + linreg_10[2])
residuum_10 = druck_11_ln - (linreg_11[0] * temp_11_T + linreg_11[2])
residuum_11 = druck_12_ln - (linreg_12[0] * temp_12_T + linreg_12[2])
residuum_12 = druck_13_ln -(linreg_13[0] * temp_13_T + linreg_13[2])
residuum_13 = druck_14_ln -(linreg_14[0] * temp_14_T + linreg_14[2])
residuum_14 = druck_15_ln -(linreg_15[0] * temp_15_T + linreg_15[2])
residuum_15 = druck_16_ln -(linreg_16[0] * temp_16_T + linreg_16[2])

sig_res_0 = np.zeros(Len1)
sig_res_1 = np.zeros(Len2)
sig_res_2 = np.zeros(Len3)
sig_res_3 = np.zeros(Len4)
sig_res_4 = np.zeros(Len5)
sig_res_5 = np.zeros(Len6)
sig_res_6 = np.zeros(Len7)
sig_res_7 = np.zeros(Len8)
sig_res_8 = np.zeros(Len9)
sig_res_9 = np.zeros(Len10)
sig_res_10 = np.zeros(Len11)
sig_res_11 = np.zeros(Len12)
sig_res_12 = np.zeros(Len13)
sig_res_13 = np.zeros(Len14)
sig_res_14 = np.zeros(Len15)
sig_res_15 = np.zeros(Len16)

sig_res_0 = np.sqrt(sig_lnp_1**2 + (temp_1_T*linreg_1[1])**2 + linreg_1[3]**2)
sig_res_1 = np.sqrt(sig_lnp_2**2 + (temp_2_T*linreg_2[1])**2 + linreg_2[3]**2)
sig_res_2 = np.sqrt(sig_lnp_3**2 + (temp_3_T*linreg_3[1])**2 + linreg_3[3]**2)
sig_res_3 = np.sqrt(sig_lnp_4**2 + (temp_4_T*linreg_4[1])**2 + linreg_4[3]**2)
sig_res_4 = np.sqrt(sig_lnp_5**2 + (temp_5_T*linreg_5[1])**2 + linreg_5[3]**2)
sig_res_5 = np.sqrt(sig_lnp_6**2 + (temp_6_T*linreg_6[1])**2 + linreg_6[3]**2)
sig_res_6 = np.sqrt(sig_lnp_7**2 + (temp_7_T*linreg_7[1])**2 + linreg_7[3]**2)
sig_res_7 = np.sqrt(sig_lnp_8**2 + (temp_8_T*linreg_8[1])**2 + linreg_8[3]**2)
sig_res_8 = np.sqrt(sig_lnp_9**2 + (temp_9_T*linreg_9[1])**2 + linreg_9[3]**2)
sig_res_9 = np.sqrt(sig_lnp_10**2 + (temp_10_T*linreg_10[1])**2 + linreg_10[3]**2)
sig_res_10 = np.sqrt(sig_lnp_11**2 + (temp_11_T*linreg_11[1])**2 + linreg_11[3]**2)
sig_res_11 = np.sqrt(sig_lnp_12**2 + (temp_12_T*linreg_12[1])**2 + linreg_12[3]**2)
sig_res_12 = np.sqrt(sig_lnp_13**2 + (temp_13_T*linreg_13[1])**2 + linreg_13[3]**2)
sig_res_13 = np.sqrt(sig_lnp_14**2 + (temp_14_T*linreg_14[1])**2 + linreg_14[3]**2)
sig_res_14 = np.sqrt(sig_lnp_15**2 + (temp_15_T*linreg_15[1])**2 + linreg_15[3]**2)
sig_res_15 = np.sqrt(sig_lnp_16**2 + (temp_16_T*linreg_16[1])**2 + linreg_16[3]**2)
"""
print linreg_1

plt.figure()
#plt.errorbar(temp_1_T, residuum_0, yerr=sig_res_0, fmt=".")
#plt.errorbar(temp_2_T, residuum_1, yerr=sig_res_1, fmt=".")
plt.errorbar(temp_3_T, residuum_2, yerr=sig_res_2, fmt=".")
#plt.errorbar(temp_4_T, residuum_3, yerr=sig_res_3, fmt=".")
#plt.errorbar(temp_5_T, residuum_4, yerr=sig_res_4, fmt=".")
#plt.errorbar(temp_6_T, residuum_5, yerr=sig_res_5, fmt=".")
#plt.errorbar(temp_7_T, residuum_6, yerr=sig_res_6, fmt=".")
#plt.errorbar(temp_8_T, residuum_7, yerr=sig_res_7, fmt=".")
#plt.errorbar(temp_9_T, residuum_8, yerr=sig_res_8, fmt=".")
#plt.errorbar(temp_10_T, residuum_9, yerr=sig_res_9, fmt=".")
#plt.errorbar(temp_11_T, residuum_10, yerr=sig_res_10, fmt=".")
#plt.errorbar(temp_12_T, residuum_11, yerr=sig_res_11, fmt=".")
#plt.errorbar(temp_13_T, residuum_12, yerr=sig_res_12, fmt=".")
#plt.errorbar(temp_14_T, residuum_13, yerr=sig_res_13, fmt=".")
#plt.errorbar(temp_15_T, residuum_14, yerr=sig_res_14, fmt=".")
#plt.errorbar(temp_16_T, residuum_15, yerr=sig_res_15, fmt=".")
plt.hlines(0, 0.00273, 0.00276)
plt.xlabel('1/T in k')
plt.ylabel('Residuen in kJ/mol')
plt.show()
"""
