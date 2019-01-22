# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:09:59 2016

@author: Martin
"""
import numpy as np

f = 0.6024
w = f * 2 * np.pi
l_p = 0.6788
r = 0.04

t_a = np.zeros(3)
t_e = np.zeros(3)
n = np.zeros(3)

t_a[0] = 0.99
t_a[1] = 1.03
t_a[2] = 0.94

t_e[0] = 158.45
t_e[1] = 158.49
t_e[2] = 158.37

sig_t = 0.01/np.sqrt(12)

n[0] = 95
n[1] = 95
n[2] = 95

w = (2*np.pi*n)/(t_e-t_a)
sig_w = 2 * np.pi * np.sqrt(2) * sig_t * n / ((t_e - t_a)**2)
sig_r = 0.00005/2
sig_l = 0.01

''' Bestimmung von g '''
g = (w**2 * l_p * (1.0 + 0.5 * ((r**2) / (l_p))))
#print 'g = ', g

''' Fehlerrechnung von g '''
I_1 = ((2.0 * w * l_p * (1.0 + 0.5 * ((r**2) / (l_p**2))))**2)
I_2 = ((w**2 * ((r) / (l_p)))**2)
I_3 = ((w**2 * (1.0 - ((r**2) / (2.0 * l_p**2))))**2)

sig_g = np.sqrt(I_1*sig_w**2+I_2*sig_r**2+I_3*sig_l**2)
#print '+- ', sig_g

''' gewichteter Mittelwert von g '''
zahler = g[0]/sig_g[0]**2 + g[1]/sig_g[1]**2 + g[2]/sig_g[2]**2
nenner = 1/sig_g[0]**2 + 1/sig_g[1]**2 + 1/sig_g[2]**2

g_bar = zahler / nenner
print 'g_bar ', g_bar

sig_g_bar = np.sqrt(1/(nenner))
print '+- ', sig_g_bar
