# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:28:58 2017

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt

params=[  1.69672271436,   5.38572409169e-15,   3.71027999889e-28]
error_params=[0.000477156000104,  1.38455715946e-16,  1.57378140281e-29]
rauschen_rot=0.00028501107336930125
ablese_fehler=1./60.*2*np.pi/360.
print(ablese_fehler)
def umrechnung(wert):
    wert=np.array(wert)
    new_wert=wert/60.
    return(new_wert)
def umrechung_1(wert):
    s=2*np.pi/360
    new_wert=wert*s
    return(new_wert)
epsilon=umrechung_1(60)

blau1_l=umrechung_1(umrechnung([0,1,1])+152)
blau2_l=umrechung_1(umrechnung([53,51,51])+151)
blau3_l=umrechung_1(umrechnung([34,33,35])+151)
rot_l=umrechung_1(umrechnung([43,42,42])+148)
name_l=[blau1_l,blau2_l,blau3_l,rot_l]

blau1_r=umrechung_1(umrechnung([32,31,31])+27)
blau2_r=umrechung_1(umrechnung([39,39,41])+27)
blau3_r=umrechung_1(umrechnung([56,55,56])+27)
rot_r=umrechung_1(umrechnung([49,48,49])+30)
name_r=[blau1_r,blau2_r,blau3_r,rot_r]

delta_l=[]
error_delta_l=[]
for i in name_l:
    delta_l+=[np.mean(i)]
    error_delta_l+=[np.sqrt((0.0166666666*np.pi*2/(360*np.sqrt(12)))**2+(rauschen_rot/np.sqrt(3))**2)]

delta_r=[]
error_delta_r=[]
for i in name_r:
    delta_r+=[np.mean(i)]
    error_delta_r+=[np.sqrt((0.0166666666*np.pi*2/(360*np.sqrt(12)))**2+(rauschen_rot/np.sqrt(3))**2)]

print('1',np.array(delta_l)*360/2/np.pi)
print(np.array(error_delta_l)*360/2/np.pi)
print('2',np.array(delta_r)*360/2/np.pi)
print(np.array(error_delta_r)*360/2/np.pi)
delta=(np.array(delta_l)-np.array(delta_r))/2
error_delta=np.sqrt(np.array(error_delta_l)**2+np.array(error_delta_r)**2)/2
print('3',delta*360/2/np.pi)
print(error_delta*360/2/np.pi)
n=np.sin((delta+epsilon)/2)/np.sin(epsilon/2)
error_n=np.cos((epsilon+delta)/2)/(2*np.sin(epsilon/2))*error_delta
print('4',n)
print(error_n)


def lambd(n,error_n,params):
    lambdaq_plus=(params[0]*params[1]+np.sqrt(params[0]**2*params[1]**2 +4*params[0]*params[2]*(n-params[0])))/(2*(n-params[0]))
    lambdaq_minus=(params[0]*params[1]-np.sqrt(params[0]**2*params[1]**2 +4*params[0]*params[2]*(n-params[0])))/(2*(n-params[0]))
    #nur lamda plus reel
    error_lambda_plus=np.sqrt((error_n*(params[0]*(params[1]*np.sqrt(params[0]*(params[0]*params[1]**2- 4*params[0]*params[2] + 4*params[2]*n))+params[0]*(params[1]**2 - 2*params[2]) + 2*params[2]*n))/(2*np.sqrt(2)*(params[0] - n)**2*np.sqrt(params[0]*(params[0]*(params[1]**2 - 4*params[2])+4*params[2]*n))*np.sqrt(-(np.sqrt(params[0]*(params[0]*params[1]**2 - 4*params[0]*params[2] + 4*params[2]*n)) + params[0]*params[1])/(params[0] - n))))**2
                              +(error_params[0]*(n*(params[1]*np.sqrt(params[0]*(params[0]*params[1]**2-4*params[0]*params[2]+4*params[2]*n))+params[0]*params[1]**2-2*params[0]*params[2]+2*params[2]*n))/(2*(params[0]-n)**2*np.sqrt(params[0]*(params[0]*params[1]**2-4*params[0]*params[2]+4*params[2]*n))))**2
                              +(error_params[1]*(params[0]*((params[0]*params[1])/np.sqrt(params[0]*(params[0]*(params[1]**2-4*params[2])+4*params[2]*n))+1))/(2*(n-params[0])))**2
                              +(error_params[2]*(params[0]/np.sqrt(params[0]*(params[0]*params[1]**2-4*params[0]*params[2]+4*params[2]*n))))**2)
    return (lambdaq_plus,lambdaq_minus,error_lambda_plus)
lambda_=[]
lambda_theo=np.array([468.01,472.22,481.05,636.23])*10**(-9)
error_lambda=[]
#name=[blau1,blau2,blau3,rot]
for i in range(4):
    s=lambd(n[i],error_n[i],params)
    if s[0]>0:
        lambda_+=[np.sqrt(s[0])]
    elif  s[1]>0:
        lambda_+=[np.sqrt(s[1])]
    error_lambda+=[lambd(n[i],error_n[i],params)[2]]
print('lambda',np.array(lambda_)*10**9)
print('error_lambda',np.array(error_lambda)*10**9)
standardabweichung=[]
for i in range(4):
    standardabweichung+=[abs(lambda_[i]-lambda_theo[i])/error_lambda[i]]
print(standardabweichung)
