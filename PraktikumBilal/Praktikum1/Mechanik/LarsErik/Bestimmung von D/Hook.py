# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import Praktikum as p
import matplotlib.pyplot as plt

#Zellen einzeln ausführen!!!

"Hooksche Feder"
"m*g=-Dx"


"Rohdaten"
x0=22.8*10**(-2)
sig_x0=10**(-3) #~1mm Ablesefehler in beide Richtungen (geschaetzt)
m_i=np.array([0.1205,0.1003,0.0503,0.0200])
sig_m=np.array([0.0001,0.0001,0.0001,0.0001])/np.sqrt(12)
x_strich=np.array([0.45,0.411,0.317,0.261])
sig_x_strich=sig_x0
x=x_strich-x0
sig_x= np.sqrt(2)*sig_x0
g=9.81

"Plot"
plt.errorbar(-x,m_i*g,xerr=sig_x ,yerr=sig_m*g, fmt=".")
plt.xlabel('-x')
plt.ylabel('m*g')
plt.title("Bestimmung von D (m*g=-Dx)")

"Lineare Regression und mit in den Plot"
data=p.lineare_regression_xy(-x,m_i*g,sig_x,sig_m*g)
print data
gerade=data[0]*(-x)+data[2]
plt.plot(-x,gerade)
#%%
"Residuen"

res=m_i*g-gerade
sig_res=np.sqrt((sig_m*g)**2+(x*data[1])**2+(data[3])**2)
plt.errorbar(-x,res,yerr=sig_res, fmt=".")
plt.xlabel('-x')
plt.ylabel('Residuen')
plt.hlines(0,-0.25,0)
plt.title("Residuenplot mit Fehlern")

