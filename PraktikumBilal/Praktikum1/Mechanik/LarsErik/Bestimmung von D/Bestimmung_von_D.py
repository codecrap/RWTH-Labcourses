# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:16:07 2016

@author: Erik
"""
#Hook nicht vorher ausführen

import numpy as np
import matplotlib.pyplot as plt
import rohdaten
import kappa
import Praktikum as p

"Bestimmung von D aus Kopplungskonstante~1+steigung*1/(l_f^2)"


"""D=m*l_s*g/data[1]"""
"""g=9.81"""
g=g_bar2
sig_g= sig_g_bar2

l_f=np.array([l_2,l_3,l_4,l_6,l_7,l_8])+stueck #1. Wert raus, da kappa nicht bestimmbar
print m
sig_1_durch_k=sig_k/(k**2)
sig_1_durch_l_f2=2*sig_massband*np.array([1,1,1,1,1,1])/(l_f**3)
"Plot"
plt.errorbar(1/(l_f**2),1/k,xerr=sig_1_durch_l_f2 ,yerr=sig_1_durch_k, fmt=".")
plt.xlabel('1/(l_f)^2')
plt.ylabel('1/k')
plt.title("Bestimmung von D aus Steigung")


"Lineare Regression und mit in den Plot"
data=p.lineare_regression_xy(1/(l_f**2),1/k,sig_1_durch_l_f2 ,sig_1_durch_k)
print data
gerade=data[0]*(1/(l_f**2))+data[2]
plt.plot(1/(l_f**2),gerade)

#%%

"Residuen"

res=1/k-gerade
sig_res=np.sqrt((sig_1_durch_k)**2+(1/(l_f**2)*data[1])**2+(data[3])**2)
print"sig_res= ", sig_res
plt.errorbar(1/(l_f**2),res,yerr=sig_res, fmt=".")
plt.xlabel('1/(l_f**2)')
plt.ylabel('Residuen')
plt.hlines(0,0,14)
plt.title("Residuenplot mit Fehlern")

#%%
D=m*l_s*g/data[0]
print "D= ",D
sig_D=m*np.sqrt((g*sig_ls/(data[0]))**2+(g*l_s*data[1]/((data[0])**2))**2+(l_s*sig_g/(data[0]))**2)
print "sig_D= ",sig_D

#print m,l_s,g,data[0]
#print sig_ls,data[0],data[1],sig_g