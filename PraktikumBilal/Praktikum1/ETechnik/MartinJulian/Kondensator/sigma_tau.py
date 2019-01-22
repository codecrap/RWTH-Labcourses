# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:44:08 2016

@author: Apoca1yptica
"""

import Rohdaten_C_Oszi
import numpy as np
import matplotlib.pyplot as plt


sig_tau_ent_1=np.sqrt((np.sqrt(2)*sig_T/np.log(U1_1/U1_2))**2+(delta_t_e1*sig_U/U1_1)**2+(delta_t_e1*sig_U/U1_2)**2)
sig_tau_ent_2=np.sqrt((np.sqrt(2)*sig_T/np.log(U2_1/U2_2))**2+(delta_t_e1*sig_U/U2_1)**2+(delta_t_e1*sig_U/U2_2)**2)
sig_tau_ent_3=np.sqrt((np.sqrt(2)*sig_T/np.log(U3_1/U3_2))**2+(delta_t_e1*sig_U/U3_1)**2+(delta_t_e1*sig_U/U3_2)**2)
sig_tau_ent_4=np.sqrt((np.sqrt(2)*sig_T/np.log(U4_1/U4_2))**2+(delta_t_e1*sig_U/U4_1)**2+(delta_t_e1*sig_U/U4_2)**2)

sig_tau_auf_1=np.sqrt((np.sqrt(2)*sig_T/np.log(I1_1/I1_2))**2+(delta_t_e1*sig_I/I1_1)**2+(delta_t_e1*sig_I/I1_2)**2)
sig_tau_auf_2=np.sqrt((np.sqrt(2)*sig_T/np.log(I2_1/I2_2))**2+(delta_t_e1*sig_I/I2_1)**2+(delta_t_e1*sig_I/I2_2)**2)
sig_tau_auf_3=np.sqrt((np.sqrt(2)*sig_T/np.log(I3_1/I3_2))**2+(delta_t_e1*sig_I/I3_1)**2+(delta_t_e1*sig_I/I3_2)**2)
sig_tau_auf_4=np.sqrt((np.sqrt(2)*sig_T/np.log(I4_1/I4_2))**2+(delta_t_e1*sig_I/I4_1)**2+(delta_t_e1*sig_I/I4_2)**2)



print sig_tau_auf_1
print sig_tau_auf_2
print sig_tau_auf_3
print sig_tau_auf_4

print sig_tau_ent_1
print sig_tau_ent_2
print sig_tau_ent_3
print sig_tau_ent_4

tau_auf_1=delta_t_e1/np.log(I1_1/I1_2)
tau_auf_2=delta_t_e1/np.log(I2_1/I2_2)
tau_auf_3=delta_t_e1/np.log(I3_1/I3_2)
tau_auf_4=delta_t_e1/np.log(I4_1/I4_2)

tau_ent_1=delta_t_e1/np.log(U1_1/U1_2)
tau_ent_2=delta_t_e1/np.log(U2_1/U2_2)
tau_ent_3=delta_t_e1/np.log(U3_1/U3_2)
tau_ent_4=delta_t_e1/np.log(U4_1/U4_2)

rel_sig_tau_auf_1=sig_tau_auf_1/tau_auf_1
rel_sig_tau_auf_2=sig_tau_auf_2/tau_auf_2
rel_sig_tau_auf_3=sig_tau_auf_3/tau_auf_3
rel_sig_tau_auf_4=sig_tau_auf_4/tau_auf_4

rel_sig_tau_ent_1=sig_tau_ent_1/tau_ent_1
rel_sig_tau_ent_2=sig_tau_ent_2/tau_ent_2
rel_sig_tau_ent_3=sig_tau_ent_3/tau_ent_3
rel_sig_tau_ent_4=sig_tau_ent_4/tau_ent_4


print rel_sig_tau_auf_1
print rel_sig_tau_auf_2
print rel_sig_tau_auf_3
print rel_sig_tau_auf_4

print rel_sig_tau_auf_1
print rel_sig_tau_auf_2
print rel_sig_tau_auf_3
print rel_sig_tau_auf_4


sig_C_auf_1=np.sqrt((sig_tau_auf_1/R)**2+(tau_auf_1*sig_R/R**2)**2)
sig_C_auf_2=np.sqrt((sig_tau_auf_2/R)**2+(tau_auf_2*sig_R/R**2)**2)
sig_C_auf_3=np.sqrt((sig_tau_auf_3/R)**2+(tau_auf_3*sig_R/R**2)**2)
sig_C_auf_4=np.sqrt((sig_tau_auf_4/R)**2+(tau_auf_4*sig_R/R**2)**2)

sig_C_ent_1=np.sqrt((sig_tau_ent_1/R)**2+(tau_ent_1*sig_R/R**2)**2)
sig_C_ent_2=np.sqrt((sig_tau_ent_2/R)**2+(tau_ent_2*sig_R/R**2)**2)
sig_C_ent_3=np.sqrt((sig_tau_ent_3/R)**2+(tau_ent_3*sig_R/R**2)**2)
sig_C_ent_4=np.sqrt((sig_tau_ent_4/R)**2+(tau_ent_4*sig_R/R**2)**2)

print sig_C_auf_1
print sig_C_auf_2
print sig_C_auf_3
print sig_C_auf_4
print
print sig_C_ent_1
print sig_C_ent_2
print sig_C_ent_3
print sig_C_ent_4

C_auf_1=tau_auf_1/R
C_auf_2=tau_auf_2/R
C_auf_3=tau_auf_3/R
C_auf_4=tau_auf_4/R

C_ent_1=tau_ent_1/R
C_ent_2=tau_ent_2/R
C_ent_3=tau_ent_3/R
C_ent_4=tau_ent_4/R

print
print C_auf_1
print C_auf_2
print C_auf_3
print C_auf_4
print
print C_ent_1
print C_ent_2
print C_ent_3
print C_ent_4
print

C_auf=np.array([C_auf_1,C_auf_2,C_auf_3,C_auf_4,])
C_ent=np.array([C_ent_1,C_ent_2,C_ent_3,C_ent_4,])
sig_C_auf=np.array([sig_C_auf_1,sig_C_auf_2,sig_C_auf_3,sig_C_auf_4,])
sig_C_ent=np.array([sig_C_ent_1,sig_C_ent_2,sig_C_ent_3,sig_C_ent_4,])

C_ges=np.concatenate((C_auf,C_ent))
sig_C_ges=np.concatenate((sig_C_auf,sig_C_ent))
sig_C_ges_sysstat=sig_C_ges+(59.3376/np.sqrt(48)/988.5645)*C_ges
sig_C_ges_sys=(59.3376/np.sqrt(48)/988.5645)*C_ges
"""
M_sig_C_auf=np.sqrt(1/np.sum(sig_C_auf**(-2)))
M_C_auf=np.sum(C_auf/(sig_C_auf)**2)*M_sig_C_auf**2

M_sig_C_ent=np.sqrt(1/np.sum(sig_C_ent**(-2)))
M_C_ent=np.sum(C_ent/(sig_C_ent)**2)*M_sig_C_ent**2
print M_C_auf,"+/-", M_sig_C_auf
print M_C_ent,"+/-", M_sig_C_ent
"""






'gewichteter Mittelwert'
os_sig_C1=1/np.square(sig_C_auf)
sig_C1_mean=np.sqrt(1/(np.sum(os_sig_C1)))
print 'Fehler des gewichteten Mittelwerts', sig_C1_mean
C1_mean_weighted=sig_C1_mean**2*(np.sum(C_auf*os_sig_C1))
print 'gewichteter Mittelwert', C1_mean_weighted

'gewichteter Mittelwert'
os_sig_C2=1/np.square(sig_C_ent)
sig_C2_mean=np.sqrt(1/(np.sum(os_sig_C2)))
print 'Fehler des gewichteten Mittelwerts', sig_C2_mean
C2_mean_weighted=sig_C2_mean**2*(np.sum(C_ent*os_sig_C2))
print 'gewichteter Mittelwert', C2_mean_weighted


'gewichteter Mittelwert'
os_sig_C_ges=1/np.square(sig_C_ges)
sig_C_ges_mean=np.sqrt(1/(np.sum(os_sig_C_ges)))
print 'Fehler des gewichteten Mittelwerts', sig_C_ges_mean
C_ges_mean_weighted=sig_C_ges_mean**2*(np.sum(C_ges*os_sig_C_ges))
print 'gewichteter Mittelwert', C_ges_mean_weighted


'gewichteter Mittelwert'
os_sig_C_ges_sysstat=1/np.square(sig_C_ges_sysstat)
sig_C_ges_sysstat_mean=np.sqrt(1/(np.sum(os_sig_C_ges_sysstat)))
print 'Fehler des gewichteten Mittelwerts', sig_C_ges_sysstat_mean
C_ges_sysstat_mean_weighted=sig_C_ges_sysstat_mean**2*(np.sum(C_ges*os_sig_C_ges_sysstat))
print 'gewichteter Mittelwert', C_ges_sysstat_mean_weighted


'gewichteter Mittelwert'
os_sig_C_ges_sys=1/np.square(sig_C_ges_sys)
sig_C_ges_sys_mean=np.sqrt(1/(np.sum(os_sig_C_ges_sys)))
print 'Fehler des gewichteten Mittelwerts', sig_C_ges_sys_mean
C_ges_sys_mean_weighted=sig_C_ges_sys_mean**2*(np.sum(C_ges*os_sig_C_ges_sys))
print 'gewichteter Mittelwert', C_ges_sys_mean_weighted


#%%
'Grafische Darstellung der Ergebnisse'
'Aufladung'
y=C_auf
y_error=sig_C_auf
mean=C1_mean_weighted
sig_mean=sig_C1_mean
N=len(y)
n=np.linspace(0,N-1,N)
print n
print y
plt.errorbar(n,y,yerr=y_error,fmt='.')
plt.hlines(mean,-1,N+1,label='Gewichteter Mittelwert='+str(mean)+'+/-'+str(sig_mean))
plt.xlim(-1,N+1)
plt.xlabel('Messpunkt')
plt.ylabel('Kapazitaet')
plt.title('Gewichteter Mittelwert vs. Einzelwerte: Aufladung des Kondensators')
plt.legend()

#%%
'Entladung'
y=C_ent
y_error=sig_C_ent
mean=C2_mean_weighted
sig_mean=sig_C2_mean
N=len(y)
n=np.linspace(0,N-1,N)
print n
print y
plt.errorbar(n,y,yerr=y_error,fmt='.')
plt.hlines(mean,-1,N+1,label='Gewichteter Mittelwert='+str(mean)+'+/-'+str(stat)+'+/-'+str(sig_mean))
plt.xlim(-1,N+1)
plt.xlabel('Messpunkt')
plt.ylabel('Kapazitaet')
plt.title('Gewichteter Mittelwert vs. Einzelwerte: Entladung des Kondensators')
plt.legend()

#%%
#%%
'Aufladung und Entladung'
y=C_ges
y_error=sig_C_ges_sysstat
mean=C_ges_sysstat_mean_weighted
N=len(y)
n=np.linspace(0,N-1,N)
print n
print y
plt.errorbar(n,y,yerr=y_error,fmt='.')
plt.hlines(mean,-1,N+1,label='Gewichteter Mittelwert='+str(mean)+'+/-'+str(sig_C_ges_mean)+'+/-'+str(sig_C_ges_sys_mean)+'F')
plt.xlim(-1,N+1)
plt.xlabel('Messpunkt')
plt.ylabel('Kapazitaet')
plt.title('Gewichteter Mittelwert vs. Einzelwerte: Aufladung und Entladung des Kondensators')
plt.legend()






