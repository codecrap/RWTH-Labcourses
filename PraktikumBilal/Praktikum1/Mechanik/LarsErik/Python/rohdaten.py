# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:13:29 2016

@author: lars
"""
'''Rohdaten Lars und Erik'''
'Massen in kg'
m1=1.0217
m2=1.0212
m=(m1+m2)/2

sig_m= 0.1*10**(-3)

'Längen in m'
sig_schieblehre=0.05*10**(-3)
sig_massband=1*10**(-2) ################### evtl neu abschätzen 

d=8.4*10**(-2)
r=d/2

sig_d=sig_schieblehre
sig_r=sig_d/2

stueck=3.2*10**(-2)
l_p_strich=61.5*10**(-2)

l_p=l_p_strich+r+stueck

###########################
'Stange'
l_st=100.2*10**(-2)
#Lochstellen
l_1=13.2*10**(-2)
l_2=25.3*10**(-2)
l_3=37.8*10**(-2)
l_4=50.3*10**(-2)
#l_5 von Pendelkörper besetzt
l_6=75.3*10**(-2)
l_7=87.3*10**(-2)
l_8=99.4*10**(-2)
#Abstand Schwerpunkt - 
l_s=l_p