# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:07:43 2016

@author: lars
"""
import numpy as np
w_s_1 = np.array([0.6040,0.6034,0.6034,0.60178,0.6010,0.5992])*2*np.pi
w_s_schraeg_hoch = np.array([0.6012,0.6026,0.6026,0.5994,0.5990,0.5985])*2*np.pi
w_s_schraeg_runter= np.array([0.6071,0.6047,0.6047,0.6037,0.6033,0.6000])*2*np.pi
w_sf_1 = np.array([0.6367,0.6685,0.7105,0.8084,0.8584,0.9075])*2*np.pi
w_sf_schraeg_hoch = np.array([0.6351,0.6663,0.7086,0.8064,0.8560,0.9051])*2*np.pi
w_sf_schraeg_runter = np.array([0.6379,0.6707,0.7129,0.8107,0.8603,0.9092])*2*np.pi

'omega_s und omega_sf'
w_s=(w_s_1+w_s_schraeg_hoch+w_s_schraeg_runter)/3
sig_w_s=np.abs((w_s_1-w_s)/2)
w_sf=(w_sf_1+w_sf_schraeg_hoch+w_sf_schraeg_runter)/3
sig_w_sf=np.abs((w_sf_1-w_sf)/2)

'omega_s und omega_sf'
"""
w_sf=w_sch+w_k
w_s=w_k-w_sch
"""
print "w_s= ",w_s
print "w_sf= ",w_sf

print "sig_w_s= ",sig_w_s
print "sig_w_sf= ",sig_w_sf


k=(w_sf**2-w_s**2)/(w_sf**2+w_s**2)

sig_k_sf=sig_w_sf*(2*w_sf*(w_sf**2+w_s**2)**(-1)-(w_sf**2-w_s**2)*2*w_sf/((w_sf**2+w_s**2)**2))
sig_k_s=sig_w_s*(-2*w_s/(w_sf**2-w_s**2)-2*w_s*(w_sf**2-w_s**2)/((w_sf**2+w_s**2)**2))
sig_k=np.sqrt(sig_k_sf**2+sig_k_s**2)

print "kappa= ", k
print "sig_k= ", sig_k

#%%
"Gegensinnig"

w_sf=(0.6136+0.6114+0.6158)/3
sig_w_sf=np.abs((0.6136-w_sf)/2)

"""Gleichsinnig"""
w_s=(0.6027+0.6022+0.6041)/3
sig_w_s=np.abs((0.6027-w_s)/2)