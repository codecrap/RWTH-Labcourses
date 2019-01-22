# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:37:17 2016

@author: Erik
"""
'Wichtig: für Plots die Zellen einzeln ausfuehren!!!'
import numpy as np
import Praktikum as p
import matplotlib.pyplot as plt
R=np.zeros(4)
sigR=np.zeros(4)
sig_delta=np.zeros(4)
delta=np.zeros(4)

#Messung 1
R[0]=9.5+2.4
sigR[0]=0.05*R[0] #5%*R
sig_delta[0]=0.653738475824
delta[0]=175.022513197

#Messung 2
R[1]=9.5+0.02
sigR[1]=0.05*R[1] #5%*R
sig_delta[1]=0.527
delta[1]=150.997

#Messung 3
R[2]=9.5+5.5
sigR[2]=0.05*R[2] #5%*R
sig_delta[2]=1.0497
delta[2]=225.027

#Messung 4
R[3]=9.5+11.8
sigR[3]=0.05*R[3] #5%*R
sig_delta[3]=1.552
delta[3]=285.786
#print R,'\n',sigR,'\n',sig_delta,'\n',delta

#%%
'lineare Regression'

data=p.lineare_regression_xy(R,delta,sigR,sig_delta)
print data
L=1/(2*data[0])
print "L=", L
sigL=data[1]/(2*data[0]**2)
print 'Fehler auf L=',sigL
print "Relativer Fehler=", sigL/L

chi2_n=data[4]/2
print chi2_n

x=np.linspace(8,25)
plt.errorbar(R,delta,sigR,sig_delta,fmt='.')
plt.plot(x,x*data[0]+data[2],label='delta=A*R+B, A=12.087, B=35.297,  \n L=1/(2*A)=0.0414 +/- 0.0033 Henry \n chi^2/f=0.738')
plt.xlabel('R')
plt.ylabel('delta')
plt.title('Bestimmung der Induktivitaet durch lineare Regression')
plt.legend()
plt.show()
#%%
x1=[1,2,3,4]
'Residuen'
res=delta-R*data[0]-data[2]
'Fehler aufs Residuum'
sigf=np.sqrt(np.square(x1)*data[1]**2+data[3]**2)
print sigf
sigres=np.sqrt(np.square(sig_delta)+np.square(sigf))
#print res

plt.errorbar(x1,res,yerr=sigres,fmt='.')
plt.hlines(0,-1,5)
plt.xlim(-1,5)
plt.xlabel('Messwert')
plt.ylabel('Residuum')
plt.title('Residuenplot')