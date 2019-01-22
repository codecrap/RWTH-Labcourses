# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:28:59 2016

@author: lars
"""
import einlesen
import matplotlib.pyplot as plt
import os 
import numpy as np
import Praktikum

p = einlesen.PraktLib('heizen.lab','cassy')
data = p.getdata()
messpunkt = data[:,0]
zeit = data[:,1]
druck = data[:,2]
T1 = data [:,3]
T1=T1[7838:]

print len(T1)
"""
####
plt.figure()
plt.plot(zeit,druck)
#plt.title(csv)
plt.xlabel('t in ms')
plt.ylabel('p in hPa')
plt.show()
"""
T1_mean = T1.mean()


n = len(T1)
sig_T1=np.sqrt(np.sum((T1-T1_mean)**2)/(n-1))
sig_T1_mean=sig_T1/np.sqrt(n)

print 'Fixpunkt bei 100 grad'
print 'Mittelwert T',T1_mean, 'Fehler auf die einzel T', sig_T1, 'Fehler auf das Mittel T', sig_T1_mean

#%%

p = einlesen.PraktLib('temp_k_3.lab','cassy')
data = p.getdata()
messpunkt = data[:,0]
zeit = data[:,1]
druck = data[:,2]
T = data [:,3]

print len(T)
"""
####
plt.figure()
plt.plot(zeit,druck)
#plt.title(csv)
plt.xlabel('t in ms')
plt.ylabel('p in hPa')
plt.show()
"""
T_mean = T.mean()


n = len(T)
sig_T=np.sqrt(np.sum((T-T_mean)**2)/(n-1))
sig_T_mean=sig_T/np.sqrt(n)

print 'Fixpunkt bei 0 grad'
print 'Mittelwert T',T_mean, 'Fehler auf die einzel T', sig_T, 'Fehler auf das Mittel T', sig_T_mean


T_T_mean=(T1_mean+T_mean)/2

fixpunkte_y=np.array([T_mean,T1_mean])-T_T_mean
fixpunkte_x=np.array([273.16,273.16+99.4]) #T_R

sig_fixpunkte_y=np.array([sig_T_mean,sig_T1_mean])
plt.errorbar(fixpunkte_x,fixpunkte_y,yerr=sig_fixpunkte_y,fmt='.')

lin_reg=Praktikum.lineare_regression(fixpunkte_x,fixpunkte_y,sig_fixpunkte_y)

print lin_reg

plt.plot(fixpunkte_x,lin_reg[0]*fixpunkte_x+lin_reg[2])
plt.title('(T-T_m)=a_2(T_R)+b_2')
plt.hlines(0,260,380)
a_2 = lin_reg[0]
sig_a_2=lin_reg[1]

b_2 = lin_reg[2]
sig_b_2 = lin_reg[3]

#%%

a_1 = 1/a_2
b_1 = -b_2/a_2
print b_1

plt.plot(fixpunkte_y,fixpunkte_y*a_1+b_1)
plt.hlines(b_1,-60,60)
plt.vlines(0,260,380)
plt.title('T_R=a_1(T-T_m)+b_1')
plt.xlabel('T-T_m')
plt.ylabel('T_R')

sig_a_1 = sig_a_2/((a_2)**2)
sig_b_1 = b_1*np.sqrt((sig_b_2/b_2)**2+(sig_a_2/a_2)**2) 

sig_T_R = np.sqrt((fixpunkte_y[0]*sig_a_1)**2+(sig_b_1)**2)

print sig_T_R