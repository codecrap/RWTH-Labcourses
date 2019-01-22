# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:52:45 2016

@author: lars
"""

import numpy as np

'FFT'
sig_f_fft=0.1
f_min=141.3
f_plus=147.0

f_res = (f_min+f_plus)/2
f_sch = (f_plus-f_min)/2

sig_f_res = 2**(-0.5)*sig_f_fft
sig_f_sch = sig_f_res

print f_res, sig_f_res
print f_sch, sig_f_sch

'Nullstellen'
t_1=378.2*10**(-3)
t_2=533.4*10**(-3)
t_3=696.1*10**(-3)

sig_t=1*10**(-3) #s

n=22+23

f_res_null=n/(t_3-t_1)
sig_f_res_null=np.sqrt(2)*n*sig_t/((t_3-t_1)**2)

print f_res_null, sig_f_res_null

t_1=191.8*10**(-3)
t_2=4382.2*10**(-3)

sig_t=10**(-2)
n=12

f_sch_null=n/(t_2-t_1)
sig_f_sch_null=np.sqrt(2)*n*sig_t/((t_2-t_1)**2)

print f_sch_null, sig_f_sch_null