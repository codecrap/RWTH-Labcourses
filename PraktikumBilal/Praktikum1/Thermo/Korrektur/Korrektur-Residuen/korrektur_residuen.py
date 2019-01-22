# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import hauptmessung_le
import Praktikum as P
import einlesen
import numpy as np
import matplotlib.pyplot as plt

#print 'druck', druck
#print temperatur


'Hauptmessung also Gruppe 1'
p=druck_5
T=temp_5
print p[0]
print T[0]

lnp=np.log(p)
rez_T=1/T

#print lnp[0]
#print rez_T[0]

#Gruppe 1
#sig_p = 0.370 * np.ones(len(p))
#sig_T = 0.054 * np.ones(len(T))

#Gruppe 2
sig_p = 0.347 * np.ones(len(p))
sig_T = 0.069 * np.ones(len(T))


sig_lnp = sig_p/p
sig_rez_T = sig_T*T**(-2)

data=P.lineare_regression_xy(rez_T,lnp,sig_rez_T,sig_lnp)
print data[4]
print data[4]/(len(p)-2)
#%%
'Lineare Regression'
a_string = 'a = '+str(round(data[0],3))+'+/-'+str(round(data[1],3))
b_string = 'b = '+str(round(data[2],3))+'+/-'+str(round(data[3],3))
chi_string = 'chi**2/f = '+str(round(data[4]/(len(p)-2),3))

plt.errorbar(rez_T,lnp,xerr=sig_rez_T,yerr=sig_lnp,fmt='.',label=a_string+'\n'+b_string+'\n'+chi_string) ##########
plt.plot(rez_T,rez_T*data[0]+data[2])
plt.xlabel('1/T in 1/K',fontsize='large')
plt.ylabel('ln(p)',fontsize='large')
plt.title('Lineare Regression Gruppe 1, 1. Teilmessung 5. Intervall',fontsize='large')
plt.legend()
plt.show()

#%%
'Residuenplot'
print data
print sig_p[0]/p[0]
print data[1]*rez_T[0]
print data[3]
'Residuen'
res=lnp-rez_T*data[0]-data[2]
'Fehler aufs Residuum'
sigres=np.sqrt((sig_p/p)**2+(sig_rez_T*data[0])**2)
#sigres=np.sqrt((sig_p/p)**2)
#print res
plt.hlines(0,0.002784,0.002802)
plt.errorbar(rez_T,res,yerr=sigres,fmt='.')
plt.title('Residuenplot Gruppe 1, 1. Teilmessung, 5. Intervall',fontsize='large')
plt.xlabel('1/T in 1/K',fontsize='large')
plt.ylabel('Residuum',fontsize='large')

#%%
"""
'Residuenplot versetzt'
print data
print sig_p[0]/p[0]
print data[1]*rez_T[0]
print data[3]
x_1=np.linspace(1,len(lnp),num=len(lnp))
print x_1
'Residuen'
res=lnp-x_1*data[0]-data[2]
'Fehler aufs Residuum'
sigres=np.sqrt((sig_p/p)**2+(sig_rez_T*data[0])**2)
#sigres=np.sqrt((sig_p/p)**2)
#print res

plt.errorbar(x_1,res,yerr=sigres,fmt='.')
plt.xlabel('Messwert')
plt.ylabel('Residuum')
plt.title('Residuenplot')
"""