# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:35:19 2016

@author: lars
"""

import Rohdaten

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


'3.Messung'
f3=np.ones(len(T3)-1)
d3=np.ones(len(T3)-1)
print '\n''Messung 3'
for i in xrange(len(T3)-1):
    f=1/(T3[i+1]-T3[i])
    print 'frequenz',f
    delta=np.log(U3[i]/U3[i+1])/(T3[i+1]-T3[i])
    print 'Dämpfungskonstante', delta
    f3[i]=f
    d3[i]=delta



'4.Messung'
f4=np.ones(len(T4)-1)
d4=np.ones(len(T4)-1)    
print '\n''Messung 4'
for i in xrange(len(T4)-1):
    f=1/(T4[i+1]-T4[i])
    delta=np.log(U4[i]/U4[i+1])/(T4[i+1]-T4[i])
    f4[i]=f
    d4[i]=delta
    f4[2]=f4[2]*2                               ### 4. von 5 Werten weggelassen
    print 'frequenz',f4[i]
    print 'Dämpfungskonstante', delta

###
print
print 'Mittelwerte Messung1:','frequenz',f1.mean(),'Dämpfung',d1.mean()
print 'Mittelwerte Messung3:','frequenz',f3.mean(),'Dämpfung',d3.mean()
print 'Mittelwerte Messung4:','frequenz',f4.mean(),'Dämpfung',d4.mean()

'Mittelwerte Gesamt'
f_mean_mean=(f1.mean()+f3.mean()+f4.mean())/3
d_mean_mean=(d1.mean()+d3.mean()+d4.mean())/3

print 
print 'Frequenz', f_mean_mean ,'Theorie', f_theo
print 'Dämpfungskoeffizient',d_mean_mean, 'Theorie', delta_theo

#%%
'''''''''''''''''''''''' 
''' Fehlerrechnung '''
'''''''''''''''''''''''' 
'''Frequenz'''
#sig_f=sig_T/((T1[1]-T1[0])**2)
#sig_f_rel=sig_f/f1[0]
#print sig_f,sig_f_rel

'Messung1'
sig_f1=np.ones(len(T1)-1)
for i in xrange(len(T1)-1):
    sig_f=sig_T/((T1[i+1]-T1[i])**2)
    sig_f1[i]=sig_f
print sig_f1

'Messung3'
sig_f3=np.ones(len(T3)-1)
for i in xrange(len(T3)-1):
    sig_f=sig_T/((T3[i+1]-T3[i])**2)
    sig_f3[i]=sig_f
print sig_f3

'Messung3'
sig_f4=np.ones(len(T4)-1)
for i in xrange(len(T4)-1):
    sig_f=sig_T/((T4[i+1]-T4[i])**2)
    sig_f4[i]=sig_f
    sig_f4[2]=sig_f4[2]*4                   #### 4. von 5 Werten weggelassen hier Faktor 2**2=4
print sig_f4

'Mittelwert'
os_sig_f1=1/(sig_f1**2)
os_sig_f3=1/(sig_f3**2)
os_sig_f4=1/(sig_f4**2)

sig_f_mean=np.sqrt(1/(np.sum(os_sig_f1)+np.sum(os_sig_f3)+np.sum(os_sig_f4)))
print '\n','Frequenz'
print 'Fehler des gewichteten Mittelwerts', sig_f_mean

f_mean_weighted=sig_f_mean**2*(np.sum(f1*os_sig_f1)+np.sum(f3*os_sig_f3)+np.sum(f4*os_sig_f4))
print 'gewichteter Mittelwert', f_mean_weighted
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

'Messung3'
sig_delta3=np.ones(len(T3)-1)
for i in xrange(len(T3)-1):
    sig_delta=1/(T3[i+1]-T3[i])*np.sqrt((sig_U/U3[i])**2+(sig_U/U3[i+1])**2+(d3[i]*np.sqrt(2)*sig_T)**2)
    sig_delta3[i]=sig_delta
print sig_delta3

'Messung4'
sig_delta4=np.ones(len(T4)-1)
for i in xrange(len(T4)-1):
    sig_delta=1/(T4[i+1]-T4[i])*np.sqrt((sig_U/U4[i])**2+(sig_U/U4[i+1])**2+(d4[i]*np.sqrt(2)*sig_T)**2)
    sig_delta4[i]=sig_delta
print sig_delta4

'Mittelwert'
os_sig_delta1=1/(sig_delta1**2)
os_sig_delta3=1/(sig_delta3**2)
os_sig_delta4=1/(sig_delta4**2)

sig_delta_mean=np.sqrt(1/(np.sum(os_sig_delta1)+np.sum(os_sig_delta3)+np.sum(os_sig_delta4)))
print '\n','Dämpfung'
print 'Fehler des gewichteten Mittelwerts', sig_delta_mean

delta_mean_weighted=sig_delta_mean**2*(np.sum(d1*os_sig_delta1)+np.sum(d3*os_sig_delta3)+np.sum(d4*os_sig_delta4))
print 'gewichteter Mittelwert', delta_mean_weighted