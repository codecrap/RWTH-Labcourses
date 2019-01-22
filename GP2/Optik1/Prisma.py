# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:06:16 2017

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import sys
sys.path.append("../PraktikumPyLib/")

import Praktikumsroutinen_DW as res

def umrechnung(wert):
    wert=np.array(wert)
    new_wert=wert/60.
    return new_wert
def umrechung_1(wert):
    s=2*np.pi/360
    new_wert=wert*s
    return new_wert
epsilon=umrechung_1(60)

'''LINKS Kommata'''
du_vio_l=umrechung_1(umrechnung([2,1,3])+155)
vio_l=umrechung_1(umrechnung([21,20,20])+153)
bl1_l=umrechung_1(umrechnung([6,5,4])+152)
bl2_l=umrechung_1(umrechnung([39,37,41])+151)
bl_gruen_l=umrechung_1(umrechnung([52,51,53])+150)
gruen_l=umrechung_1(umrechnung([3,2,4])+150)
gelb_l=umrechung_1(umrechnung([34,33,31])+149)
rot_l=umrechung_1(umrechnung([40,38,38])+148)
'''RECHTS Kommata'''
du_vio_r=umrechung_1(umrechnung([32,31,33])+24)
vio_r=umrechung_1(umrechnung([13,14,14])+26)
bl1_r=umrechung_1(umrechnung([30,31,33])+27)
bl2_r=umrechung_1(umrechnung([52,52,53])+27)
bl_gruen_r=umrechung_1(umrechnung([41,41,49])+28)
gruen_r=umrechung_1(umrechnung([26,30,30])+29)
gelb_r=umrechung_1(umrechnung([1,0,2])+30)
rot_r=umrechung_1(umrechnung([51,52,50,52,51,52,50,50,53,51])+30)
'''winkel'''
delta_l=[]
error_delta_l=[]
delta_r=[]
error_delta_r=[]
error_rausch=np.std(rot_r)
#print(error_rausch,'Fehler')
name_l=[du_vio_l,vio_l,bl1_l,bl2_l,bl_gruen_l,gruen_l,gelb_l,rot_l]
name_r=[du_vio_r,vio_r,bl1_r,bl2_r,bl_gruen_r,gruen_r,gelb_r]
wavelength=np.array([404.66,435.83,467.81,479.99,508.58,546.07,576.96,643.85])*10**(-9)
#fehler auf winkel als 1´ als ablese fehler da dieser deutlich größer als statistischer fehler rauschen
for i in name_l:
    delta_l+=[np.mean(i)]
    error_delta_l+=[error_rausch/np.sqrt(3)]
for i in name_r:
    delta_r+=[np.mean(i)]
    error_delta_r+=[error_rausch/np.sqrt(3)]
delta_r+=[np.mean(rot_r)]
error_delta_r+=[error_rausch/np.sqrt(10)]
#print('delta_r',np.array(delta_r)*360/2/np.pi)
#print('error_delta_r',np.array(error_delta_r)*360/2/np.pi)
#print('delta_l',np.array(delta_l)*360/2/np.pi)
#print('error_delta_l',np.array(error_delta_l)*360/2/np.pi)

'''minimaler ablenkwinkel'''
delta=(np.array(delta_l)-np.array(delta_r))/2
error_delta=(np.sqrt(np.array(error_delta_l)**2+np.array(error_delta_r)**2))/2
#print('delta',np.array(delta)*360/2/np.pi)
print('error_delta',np.array(error_delta)*360/2/np.pi)
'''Bestimmung n'''
n=np.sin((delta+epsilon)/2)/np.sin(epsilon/2)
#print('n',n)
error_n=np.cos((epsilon+delta)/2)/(2*np.sin(epsilon/2))*error_delta
#print('error_n',error_n)
#linearer Fit
a,ea,b,eb,chiq_ndof=res.residuen((1/wavelength)**2,n,0,error_n,'[$m^{-2}$]','','$\lambda^{-2}$','n',ca=0,k=2.5*10**12,l=1.76,o=3.5*10**12,p=0.001)
#print('res',a,ea,b,eb)


func = lambda p, x: p[0]*(1 + p[1] * (x**2) + p[2] * (x**4))
errfunc=lambda p, x, n, n_error: (n-func(p, x))/n_error
x=1/wavelength
#fit mit leastsq
p0=[0,0,0]
#print(opt_param)
out = sc.leastsq(errfunc,p0,args=(x,n,error_n), full_output=1)
#print('out',out)
print ('a:',out[0][0], np.sqrt(out[1][0][0]))         #fehler als diagonal elemente der cov matrix
print ('b:',out[0][1], np.sqrt(out[1][1][1])*10**15)
print ('c:',out[0][2], np.sqrt(out[1][2][2])*10**28)
#print out[4]                                   # wenn 1,2,3, oder 4 dann wurde ne lösung gefunden
residue=n-func(out[0],x)
eres=np.sqrt(error_n**2)
chiq=(sum(((n-func(out[0],x))/error_n)**2))/5
print ('chi^2/ndof',chiq)
a_1=out[0][0]
ea_1=out[1][0][0]
b_1=out[0][1]
eb_1=out[1][1][1]
b_2=out[0][2]
eb_2=out[1][2][2]

h='a='+'1.6966'+'$\pm$'+'0.0003'
i='$b_1$='+'(5.4102 '+'$\pm$'+'0.083)$\cdot 10^{-15}$'
j='$b_2$='+'(3.6854 '+'$\pm$'+'0.095)$\cdot 10^{-28}$'
k=2.3*10**12
l=1.745

fig10,(ax10,ax20)=plt.subplots(2,1)
ax10.plot(1/wavelength**2,func(out[0],x),'-r')
ax10.errorbar(1/wavelength**2,n,error_n,0,'.g')
ax10.set_title('n gegen $ \lambda^{-2}$ und nicht linearer Fit ')
ax10.set_xlabel('$\lambda^{-2}$ in [$m^{-2}$]')
ax10.set_ylabel('n')
ax10.annotate('{0} \n {1} \n {2}'.format(h,i,j),xy=(k,l),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})

s=r'$\frac{\chi}{ndof}$='+('{0:9.4f}').format(chiq)
r=2.5*10**12
t=-0.0001
ax20.errorbar(1/wavelength**2,residue,eres,0,'.',capsize=3)
ax20.plot(1/wavelength**2, 0*wavelength,"red")
ax20.set_title('Residuenplot')
ax20.set_xlabel('$\lambda^{-2}$  in [$m^{-2}$]')
ax20.set_ylabel(r'$n-a(1+\frac{b_1}{\lambda^{2}}+\frac{b_2}{\lambda^4})$')
ax20.annotate('{0}'.format(s),xy=(r,t),fontsize=20,bbox={'facecolor':'white','alpha':0.5,'pad':4})

fig10.tight_layout()
