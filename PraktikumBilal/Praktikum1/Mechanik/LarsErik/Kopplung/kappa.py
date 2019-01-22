# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:07:43 2016

@author: lars
"""
import numpy as np
w_sch_1 = np.array([])
w_sch_schraeg_hoch = np.array([])
w_sch_schraeg_runter= np.array([])
w_k_1 = np.array([])
w_k_schraeg_hoch = np.array([])
w_k_schraeg_runter = np.array([])

'omega_sch und omega_k'
w_sch=(w_sch_1+w_sch_schraeg_hoch+w_sch_schraeg_runter)/3
sig_w_sch=(w_k_1-w_sch)/2 
w_k=(w_k_1+w_k_schraeg_hoch+w_k_schraeg_runter)/3
sig_w_k=(w_k_1-w_k)/2 

'omega_s und omega_sf'
w_sf=w_sch+w_k
w_s=w_k-w_sch

#print sig_w_sch