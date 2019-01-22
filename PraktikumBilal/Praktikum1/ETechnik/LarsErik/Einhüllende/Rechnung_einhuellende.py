# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:35:19 2016

@author: lars
"""

import Rohdaten_einhuellende

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
print 'Mittelwerte Messung1:','frequenz',f1.mean(),'Dämpfung',d1.mean()

#%%
'''''''''''''''''''''''' 
''' Fehlerrechnung '''
'''''''''''''''''''''''' 
'''Frequenz'''

'Messung1'
sig_f1=np.ones(len(T1)-1)
for i in xrange(len(T1)-1):
    sig_f=sig_T/((T1[i+1]-T1[i])**2)
    sig_f1[i]=sig_f
print sig_f1

'Mittelwert'
os_sig_f1=1/(sig_f1**2)

#%%
'''Dämpfungskonstante'''
sig_delta=1/(T1[1]-T1[0])*np.sqrt((sig_U/U1[0])**2+(sig_U/U1[1])**2+(d1[0]*np.sqrt(2)*sig_T)**2)
print sig_delta

'Messung1'
sig_delta1=np.ones(len(T1)-1)
for i in xrange(len(T1)-1):
    sig_delta=1/(T1[i+1]-T1[i])*np.sqrt((sig_U/U1[i])**2+(sig_U/U1[i+1])**2+(d1[i]*np.sqrt(2)*sig_T)**2)
    sig_delta1[i]=sig_delta
print sig_delta1

'Mittelwert'
os_sig_delta1=1/(sig_delta1**2)

