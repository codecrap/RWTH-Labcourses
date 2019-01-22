# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:42:47 2016

@author: Defender833
"""

'''Rohdaten Julian und Martin'''
'Massen in kg'
m1 = 1.0207
m2 = 1.0130
m = (m1+m2)/2

sig_m = 0.1*10**(-3)

'Längen in m'
sig_schieblehre = 0.05*10**(-3)
sig_massband = 0.01 

d = 0.08
r = d/2

sig_d = sig_schieblehre
sig_r = sig_d/2

stueck = 3.2*10**(-2)
l_p_strich = 61.5*10**(-2)

l_p = l_p_strich+r+stueck

'Stange'
l_s = 100.6*10**(-2)
#Lochstellen
l_1 = 13.7*10**(-2)
l_2 = 25.7*10**(-2)
l_3 = 38.7*10**(-2)
l_4 = 50.8*10**(-2)
#l_5 von Pendelkörper besetzt
l_6 = 75.5*10**(-2)
l_7 = 87.5*10**(-2)
l_8 = 99.5*10**(-2)
