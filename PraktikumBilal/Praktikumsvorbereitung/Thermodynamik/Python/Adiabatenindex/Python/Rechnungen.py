# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:00:35 2015

@author: lars
"""
import numpy as np
import matplotlib.pyplot as plt
"""Vorfaktor F=m/(pA^2)"""
p=101700.0
sig_p=318.0
A=0.201*10**(-3)
sig_A=0.0793*10**(-3)
m=0.0165
sig_m=0.0001

sig_F=np.sqrt((sig_m/(p*A**2))**2+(sig_p*m/p**2*A**2)**2+(sig_A*2*m/p**2*A**3)**2)
#print sig_F
#####################
#                   #
#                   #
#####################
"""Volumina"""
V_g=11.315/1000 #[m^3]
V_m=0.10713*10**(-2)
V_k=0.309*10**(-4)
sig_V=10**(-6)
"""Ausgangshoehe"""
L_0=0.6 #m
L_1=0.2
L_2=0.3
L_3=0.4
sig_L=0.001
"""Einstecktiefe"""
s_g=0.0021
s_m=0.0015
s_k=0.0022
"""Durchmesser"""
d=0.016
sig_d=0.0005
#

#
#
#
###################################
array_V=np.zeros(9)

V_g_20=V_g+np.pi*d**2*(L_0-L_1-s_g)
V_g_30=V_g+np.pi*d**2*(L_0-L_2-s_g)
V_g_40=V_g+np.pi*d**2*(L_0-L_3-s_g)

V_m_20=V_m+np.pi*d**2*(L_0-L_1-s_m)
V_m_30=V_m+np.pi*d**2*(L_0-L_2-s_m)
V_m_40=V_m+np.pi*d**2*(L_0-L_3-s_m)

V_k_20=V_k+np.pi*d**2*(L_0-L_1-s_k)
V_k_30=V_k+np.pi*d**2*(L_0-L_2-s_k)
V_k_40=V_k+np.pi*d**2*(L_0-L_3-s_k)

array_V[0]=V_k_40
array_V[1]=V_k_30
array_V[2]=V_k_20

array_V[3]=V_m_40
array_V[4]=V_m_30
array_V[5]=V_m_20

array_V[6]=V_g_40
array_V[7]=V_g_30
array_V[8]=V_g_20

print array_V
#
#
######################################################################
sig_v_g_20=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_1-s_g)/2)**2)
sig_v_g_30=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_2-s_g)/2)**2)
sig_v_g_40=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_3-s_g)/2)**2)

sig_v_m_20=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_1-s_m)/2)**2)
sig_v_m_30=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_2-s_m)/2)**2)
sig_v_m_40=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_3-s_m)/2)**2)

sig_v_k_20=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_1-s_k)/2)**2)
sig_v_k_30=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_2-s_k)/2)**2)
sig_v_k_40=np.sqrt(sig_V**2+(A*sig_d)**2+(d*np.pi*sig_d*(L_0-L_3-s_k)/2)**2)

#print V_m_20-V_m

array_sig_V=[sig_v_k_40,sig_v_k_30,sig_v_k_20,sig_v_m_40,sig_v_m_30,sig_v_m_20,sig_v_g_40,sig_v_g_30,sig_v_g_20]
#print array_sig_V

#
#
###################
'Vorfaktor'
vorfaktor=4.0156
vorfaktor_array=vorfaktor*np.ones(9)
#print vorfaktor_array
vorfaktor_fehler=0.0246
vorfaktor_fehler_array=vorfaktor_fehler*np.ones(9)
'Omega Quadrate'
omega_mean=1