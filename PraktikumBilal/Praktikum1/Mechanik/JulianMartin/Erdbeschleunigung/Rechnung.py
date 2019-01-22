# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:00:35 2016

@author: lars
"""

import rohdaten
import numpy as np


t_a=0.89
t_e=159.92
sig_t=0.01/np.sqrt(12)
n=96

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

