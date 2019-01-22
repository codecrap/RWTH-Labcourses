# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:50:20 2017

@author: Philipp
"""
import numpy as np
import sys
sys.path.append("../../PraktikumPyLib")
import Praktikum
#import Residuen
import Praktikumsroutinen2_DW as res
import matplotlib.pyplot as plt
plt.close('all')
lamb=632.8 *10**(-9)
s1=[7.56,7.63,7.7,7.77,7.83,7.89,7.96,8.02,8.09,8.16,8.22,8.29,8.34,8.41,8.47,8.55,8.61,8.67,8.74,8.8,8.87]
s2=[8.06,8.12,8.19,8.25,8.32,8.39,8.46,8.53,8.6,8.67,8.73,8.81,8.88,8.95,9.02]

dm1=[]
s_err1=[]
dm_err1=[]
for i in range(len(s1)):
    dm1.append(i*10+10)
    s_err1.append(0.01/np.sqrt(12))

dm2=[]
s_err2=[]
dm_err2=[]
for i in range(len(s2)):
    dm2.append(i*10+10)
    s_err2.append(0.01/np.sqrt(12))


s1=np.array(s1)/1000.
s_err1=np.array(s_err1)/1000.
a,ea,b,eb,chiq_ndof=res.residuen(np.array(dm1),s1,0,np.array(s_err1),' ',' [m]',r"$\Delta m$",'s',k=10,l=0.0084,o=25,p=-0.00001)
k1=lamb/2/a
k_err1=lamb/2/a**2*ea
print('kGruppe1= ',k1,' +- ',k_err1)

s2=np.array(s2)/1000
s_err2=np.array(s_err2)/1000
a,ea,b,eb,chiq_ndof=res.residuen(np.array(dm2),s2,0,np.array(s_err2),' ',' [m]',r"$\Delta m$",'s',k=10,l=0.0086,o=60,p=0.00001)
k2=lamb/2/a
k_err2=lamb/2/a**2*ea
print('kGruppe2= ',k2,' +- ',k_err2)



s_1=[7.55,7.61,7.67,7.73,7.78,7.84,7.88,7.94,7.99,8.05,8.1,8.16,8.22,8.27,8.34,8.39]
s_2=[8.05,8.1,8.15,8.2,8.26,8.32,8.38,8.43,8.49,8.55,8.61,8.68,8.74,8.8,8.85,8.91,8.97]


dm_1=[]
s_err_1=[]
dm_err_1=[]
for i in range(len(s_1)):
    dm_1.append(i*10+10)
    s_err_1.append(0.01/np.sqrt(12))

s_1=np.array(s_1)/1000
s_err_1=np.array(s_err_1)/1000
a,ea,b,eb,chiq_ndof=res.residuen(np.array(dm_1),s_1,0,np.array(s_err_1),' ',' [m]',r"$\Delta m$",'s',k=10,l=0.008,o=25,p=-0.00001)
l1=2*a*k1
l1_err=np.sqrt((2*k1*ea)**2+(2*a*k_err1)**2)
l1_errsy=2*a*k_err1
l1_errst=2*k1*ea
print('l +- statistisch +- systematisch')
print('Lambda der 1 Gruppe= ',l1,' +- ',l1_errst,' +- ',l1_errsy)


dm_2=[]
s_err_2=[]
dm_err_2=[]
for i in range(len(s_2)):
    dm_2.append(i*10+10)
    s_err_2.append(0.01/np.sqrt(12))

s_2=np.array(s_2)/1000
s_err_2=np.array(s_err_2)/1000
a,ea,b,eb,chiq_ndof=res.residuen(np.array(dm_2),s_2,0,np.array(s_err_2),' ',' [m]',r"$\Delta m$",'s',k=10,l=0.0087,o=60,p=0.00001)
l2=2*a*k2
l2_err=np.sqrt((2*k2*ea)**2+(2*a*k_err2)**2)
l2_errsy=2*a*k_err2
l2_errst=2*k2*ea
print('Lambda der 2 Gruppe= ',l2,' +- ',l2_errst,' +- ',l2_errsy)


#### CO_2 Messung
m1=[5,5.25,5.5,5.5,5,5]
m5=[4.5,4.25,5,4.75,4.75,5.25]
m6=[4.25,4.5,4,4.25,4.5,5]
L=0.01
a=2.657652*10**(-7)
b=1
ea=1.737522*10**(-9)
eb=0
P=992
nLuft=b+a*P
nLuft_err=np.sqrt(eb**2+(P*ea)**2)

dm_1=np.mean(m1)
dm_2=np.mean(m5)
dm_3=np.mean(m6)

dm_1_err=np.std(m1)/np.sqrt(6)
dm_2_err=np.std(m5)/np.sqrt(6)
dm_3_err=np.std(m6)/np.sqrt(6)

dn_1=dm_1*l2/2/0.01
dn_2=dm_2*l2/2/0.01
dn_3=dm_3*l2/2/0.01

dn_1_err=np.sqrt((l2/2/L*dm_1_err)**2+(dm_1/2/L*l2_err)**2)
dn_2_err=np.sqrt((l2/2/L*dm_2_err)**2+(dm_2/2/L*l2_err)**2)
dn_3_err=np.sqrt((l2/2/L*dm_3_err)**2+(dm_3/2/L*l2_err)**2)

dn_1_errsy=np.sqrt((dm_1/2/L*l2_err)**2)
dn_2_errsy=np.sqrt((dm_2/2/L*l2_err)**2)
dn_3_errsy=np.sqrt((dm_3/2/L*l2_err)**2)

dn_1_errst=np.sqrt((l2/2/L*dm_1_err)**2)
dn_2_errst=np.sqrt((l2/2/L*dm_2_err)**2)
dn_3_errst=np.sqrt((l2/2/L*dm_3_err)**2)

dn=np.array([dn_1,dn_2,dn_3])
dn_err=np.array([dn_1_err,dn_2_err,dn_3_err])
dn_errsys=np.array([dn_1_errsy,dn_2_errsy,dn_3_errsy])
dn_errsta=np.array([dn_1_errst,dn_2_errst,dn_3_errst])

xm,sx=Praktikum.gewichtetes_mittel(dn,dn_err)
dn_1=xm+nLuft-1
dn_1_err=np.sqrt(sx**2+nLuft_err**2)
print('deltan =(',dn_1*10000,' +- ',dn_1_err*10000,') *10^-4')
