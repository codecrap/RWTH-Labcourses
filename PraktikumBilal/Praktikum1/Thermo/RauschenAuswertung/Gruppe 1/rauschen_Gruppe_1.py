# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:28:59 2016

@author: lars
"""
import einlesen
import matplotlib.pyplot as plt
import Praktikum as P
import numpy as np

p = einlesen.PraktLib('rausch_1.lab','cassy')
data = p.getdata()
messpunkt1 = data[:,0]
zeit1 = data[:,1]
druck1 = data[:,2]
T1 = data [:,3]
T_neu1 = T1[366:len(T1)]

p = einlesen.PraktLib('rausch_2.lab','cassy')
data = p.getdata()
messpunkt2 = data[:,0]
zeit2 = data[:,1]
druck2 = data[:,2]
T2 = data [:,3]

druck=np.concatenate((druck1,druck2))
T=np.concatenate((T_neu1,T2))
druck_mean=druck.mean()

n = len(druck)
sig_druck=np.sqrt(np.sum((druck-druck_mean)**2)/(n-1))
print'Gruppe 1'
print 'Mittelwert druck',druck_mean, 'Fehler auf die einzel druck', sig_druck

T_mean = T.mean()


n = len(T)
sig_T=np.sqrt(np.sum((T-T_mean)**2)/(n-1))

print 'Mittelwert T',T_mean, 'Fehler auf die einzel T', sig_T