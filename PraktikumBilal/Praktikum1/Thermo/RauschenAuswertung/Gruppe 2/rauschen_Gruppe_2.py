# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:28:59 2016

@author: lars
"""
import einlesen
import matplotlib.pyplot as plt
import Praktikum as P
import numpy as np

p = einlesen.PraktLib('Temp_3K.lab','cassy')
data = p.getdata()
messpunkt = data[:,0]
zeit = data[:,1]
druck = data[:,2]
T = data [:,3]
####
plt.figure()
plt.plot(zeit,druck)
#plt.title(csv)
plt.xlabel('t in ms')
plt.ylabel('p in hPa')
plt.show()

druck_mean = druck.mean()


n = len(druck)
sig_druck=np.sqrt(np.sum((druck-druck_mean)**2)/(n-1))
print'Gruppe 2'
print 'Mittelwert druck',druck_mean, 'Fehler auf die einzel druck', sig_druck

T_mean = T.mean()


n = len(T)
sig_T=np.sqrt(np.sum((T-T_mean)**2)/(n-1))

print 'Mittelwert T',T_mean, 'Fehler auf die einzel T', sig_T