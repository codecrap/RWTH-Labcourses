# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

'''Rohdaten'''
'Spule'
L=36*10**(-3)
N=1000
R_L=9.5
I_m=1.25

'Kondensator'
C=10*10**(-6)
V_m=100

'Theoretische Werte'
delta_theo=R_L/(2*L)
f_theo=np.sqrt(1/(L*C)-R_L**2/(4*L**2))/(2*np.pi)

'Anfangsspannung'
U0=5.6

'Messbereiche'
#Spannung
U_B=16
#Zeit
T_B=50*10**(-3)

'Ablesefehler'
sig_U=0.08/(np.sqrt(12))
sig_T=100*10**(-6)/(np.sqrt(12))

'Offset'
#in V
off=50*10**(-3)

'1. Messung'
#Spannung in V
U1=np.array([3.12,1.76,1.04,0.56])
U1=U1-off
#Zeit in ms
T1ms=np.array([0.5,4.4,8.2,12])
#Zeit in s
T1=T1ms*10**(-3)
