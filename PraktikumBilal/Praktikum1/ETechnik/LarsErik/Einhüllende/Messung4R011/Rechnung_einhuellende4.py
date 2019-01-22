# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:35:19 2016

@author: lars
"""

import Rohdaten_einhuellende4

import numpy as np


'1.Messung'
f1=np.ones(len(T1)-1)
d1=np.ones(len(T1)-1)
print 'Messung 1'
for i in xrange(len(T1)-1):
    f=1/(T1[i+1]-T1[i])
    print 'frequenz',f
    delta=np.log(U1[i]/U1[i+1])/(T1[i+1]-T1[i])
    print 'Dämpfungskonstante', delta
    f1[i]=f
    d1[i]=delta

###
print
print 'Mittelwerte:','Frequenz',f1.mean(),'Dämpfung',d1.mean()

#%%
'''''''''''''''''''''''' 
''' Fehlerrechnung '''
'''''''''''''''''''''''' 
'''Frequenz'''
print '\n','Frequenz'
'Messung1'
sig_f1=np.ones(len(T1)-1)
for i in xrange(len(T1)-1):
    sig_f=sig_T/((T1[i+1]-T1[i])**2)
    sig_f1[i]=sig_f
print 'Einzelfehler',sig_f1

'gewichteter Mittelwert'
os_sig_f1=1/(sig_f1**2)
sig_f_mean=np.sqrt(1/(np.sum(os_sig_f1)))
print 'Fehler des gewichteten Mittelwerts', sig_f_mean
f_mean_weighted=sig_f_mean**2*(np.sum(f1*os_sig_f1))
print 'gewichteter Mittelwert', f_mean_weighted
print 'f_theo',f_theo

#%%
'''Dämpfungskonstante'''
print '\n','Dämpfung'
#sig_delta=1/(T1[1]-T1[0])*np.sqrt((sig_U/U1[0])**2+(sig_U/U1[1])**2+(d1[0]*np.sqrt(2)*sig_T)**2)
#print sig_delta

'Messung1'
sig_delta1=np.ones(len(T1)-1)
for i in xrange(len(T1)-1):
    sig_delta=1/(T1[i+1]-T1[i])*np.sqrt((sig_U/U1[i])**2+(sig_U/U1[i+1])**2+(d1[i]*np.sqrt(2)*sig_T)**2)
    sig_delta1[i]=sig_delta
print 'Einzelfehler',sig_delta1

'gewichteter Mittelwert'
os_sig_delta1=1/(sig_delta1**2)
sig_delta_mean=np.sqrt(1/(np.sum(os_sig_delta1)))

print 'Fehler des gewichteten Mittelwerts', sig_delta_mean

delta_mean_weighted=sig_delta_mean**2*(np.sum(d1*os_sig_delta1))
print 'gewichteter Mittelwert', delta_mean_weighted
print 'delta_theo',delta_theo

