# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:19:34 2016

@author: Erik
"""

import numpy as np
import Rohdaten

"Bestimmung von Rho mit Fehler"
rho_1=4*m_1/(np.pi*l_1*d_1_mean**2)
print "Rho1, sig1, sig/rho",rho_1
sig_rho_1=np.sqrt((sig_m/m_1)**2+(sig_l/l_1)**2+(2*sig_d_1/d_1_mean)**2)*rho_1
print sig_rho_1
print sig_rho_1/rho_1

rho_2=4*m_2/(np.pi*l_2*d_2_mean**2)
print "Rho2, sig2, sig/rho",rho_2
sig_rho_2=np.sqrt((sig_m/m_2)**2+(sig_l/l_2)**2+(2*sig_d_2/d_2_mean)**2)*rho_2
print sig_rho_2
print sig_rho_2/rho_2

rho_3=4*m_3/(np.pi*l_3*d_3_mean**2)
print "Rho3, sig3, sig/rho:",rho_3
sig_rho_3=np.sqrt((sig_m/m_3)**2+(sig_l/l_3)**2+(2*sig_d_3/d_3_mean)**2)*rho_3
print sig_rho_3
print sig_rho_3/rho_3

rho_4=4*m_4/(np.pi*l_4*d_4_mean**2)
print "Rho4, sig4, sig/rho:",rho_4
sig_rho_4=np.sqrt((sig_m/m_4)**2+(sig_l/l_4)**2+(2*sig_d_4/d_4_mean)**2)*rho_4
print sig_rho_4
print sig_rho_4/rho_4


"Elastizitätsmodul mit Fehler"
E_1= rho_1*f_1_mean**2*4*l_1**2
print"E_1=", E_1/(10**11), "*10^11"
E_2= rho_2*f_2_mean**2*4*l_2**2
print"E_2=", E_2/(10**11), "*10^11"
E_3= rho_3*f_3_mean**2*4*l_3**2
print"E_3=", E_3/(10**11), "*10^11"
E_4= rho_4*f_4_mean**2*4*l_4**2
print "E_4=",E_4/(10**11), "*10^11"

sig_E_1=E_1*np.sqrt((sig_rho_1/rho_1)**2+(2*sig_f_1/f_1_mean)**2+(2*sig_l/l_1)**2)
sig_E_2=E_2*np.sqrt((sig_rho_2/rho_2)**2+(2*sig_f_2/f_2_mean)**2+(2*sig_l/l_2)**2)
sig_E_3=E_3*np.sqrt((sig_rho_3/rho_3)**2+(2*sig_f_3/f_3_mean)**2+(2*sig_l/l_3)**2)
sig_E_4=E_4*np.sqrt((sig_rho_4/rho_4)**2+(2*sig_f_4/f_4_mean)**2+(2*sig_l/l_4)**2)

print "sig_E_1=", sig_E_1/(10**11), "*10^11"
print "sig_E_2=", sig_E_2/(10**11), "*10^11"
print "sig_E_3=", sig_E_3/(10**11), "*10^11"
print "sig_E_4=", sig_E_4/(10**11), "*10^11"


print "v_1=",np.sqrt(E_1/rho_1), " sig_v1=", np.sqrt((sig_f_1/f_1_mean)**2+(sig_l/l_1)**2)*np.sqrt(E_1/rho_1)
print "v_2=",np.sqrt(E_2/rho_2)," sig_v2=", np.sqrt((sig_f_2/f_2_mean)**2+(sig_l/l_2)**2)*np.sqrt(E_2/rho_2)
print "v_3=",np.sqrt(E_3/rho_3)," sig_v3=", np.sqrt((sig_f_3/f_3_mean)**2+(sig_l/l_3)**2)*np.sqrt(E_3/rho_3)
print "v_4=",np.sqrt(E_4/rho_4)," sig_v4=", np.sqrt((sig_f_4/f_4_mean)**2+(sig_l/l_4)**2)*np.sqrt(E_4/rho_4)