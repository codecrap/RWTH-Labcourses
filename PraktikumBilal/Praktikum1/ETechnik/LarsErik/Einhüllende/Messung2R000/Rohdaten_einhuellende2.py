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
R_L=9.5+0.02
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
U_B=20
#Zeit in ms
T_B=40

'Ablesefehler'
sig_U=0.01/(np.sqrt(12))
sig_T=0.03*10**(-3)/(np.sqrt(12))

'Offset'
#in V
off=0.13

'1. Messung'
#Spannung in V
U1=np.array([3.26,1.88,1.09,0.68,0.45])
U1=U1-off
#Zeit in ms
T1ms=np.array([3.77,7.62,11.49,15.36,19.22])
#Zeit in s
T1=T1ms*10**(-3)
