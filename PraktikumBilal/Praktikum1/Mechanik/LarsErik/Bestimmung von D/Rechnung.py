# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:00:35 2016

@author: lars
"""

import rohdaten
import numpy as np
import matplotlib.pyplot as plt


t_a=np.array([1.29,0.92,1.13,1.19])
t_e=np.array([157.69,158.98,159.16,159.2])
sig_t=0.01/np.sqrt(12)
n=np.array([94,95,95,95])

w=(2*np.pi*n)/(t_e-t_a)
sig_w=2*np.pi*np.sqrt(2)*sig_t*n/((t_e-t_a)**2)

''' Bestimmung von g '''
g = (w**2 * l_p * (1.0 + 0.5 * ((r**2) / (l_p))))
print 'g = ', g

''' Fehlerrechnung von g '''
I_1 = ((2.0 * w * l_p * (1.0 + 0.5 * ((r**2) / (l_p**2))))**2)
I_2 = ((w**2 * ((r) / (l_p)))**2)
I_3 = ((w**2 * (1.0 - ((r**2) / (2.0 * l_p**2))))**2)

sig_g = np.sqrt(I_1*sig_w**2+I_2*sig_r**2+I_3*sig_l**2)
print '+- ', sig_g

''' gewichteter Mittelwert von g '''
zahler = g[0]/sig_g[0]**2 + g[1]/sig_g[1]**2 + g[2]/sig_g[2]**2 + g[3]/sig_g[3]**2
nenner = 1/sig_g[0]**2 + 1/sig_g[1]**2 + 1/sig_g[2]**2 + 1/sig_g[3]**2

g_bar = zahler / nenner
print 'g_bar ', g_bar

sig_g_bar = np.sqrt(1/(nenner))
print '+- ', sig_g_bar

'Residuenplot'
y=g
y_error=sig_g
mean=g_bar
sig_mean=sig_g_bar
g_theo=9.81
N=len(y)
n=np.linspace(0,N-1,N)
print n
print y
plt.errorbar(n,y,yerr=y_error,fmt='.',label='Gruppe 2')
plt.hlines(mean,-1,N+1,label='Gewichteter Mittelwert (Gruppe 2)='+str(mean)+'+/-'+str(sig_mean))
plt.hlines(g_theo,-1,N+1,label='g_theo='+str(g_theo),color='r')
plt.xlim(-1,N+1)
plt.xlabel('Messpunkt')
plt.ylabel('Erdbeschleunigung')
plt.title('Gewichteter Mittelwert vs. Einzelwerte')
#plt.legend()
g_bar2=g_bar
sig_g_bar2=sig_g_bar
#%%
'Martins und Julians Daten'
g = np.array([ 9.76606978 , 9.76606978 , 9.7697922 ])
sig_g =  np.array([ 0.07172864 , 0.07172864 , 0.07175599])
g_bar =  9.76730996006
sig_g_bar = 0.0414178118927

y=g
y_error=sig_g
mean=g_bar
sig_mean=sig_g_bar
g_theo=9.81
N=len(y)
n=np.linspace(0,N-1,N)+0.5
print n
print y
plt.errorbar(n,y,yerr=y_error,fmt='.',label='Gruppe 1')
plt.hlines(mean,-1,N+1,label='Gewichteter Mittelwert (Gruppe 2)='+str(mean)+'+/-'+str(sig_mean))
plt.xlim(-1,N+1)
plt.ylim(9.6,10)
plt.xlabel('Messpunkt')
plt.ylabel('Erdbeschleunigung')
plt.title('Gewichteter Mittelwert vs. Einzelwerte')
plt.legend(fontsize='small')