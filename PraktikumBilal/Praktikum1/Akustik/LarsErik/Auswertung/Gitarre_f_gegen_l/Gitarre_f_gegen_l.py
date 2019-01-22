# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:24:12 2016

@author: lars
"""

import numpy as np
import Praktikum as p
import matplotlib.pyplot as plt

'Frequenzen in Hz'
f_0_1=110.16
f_0_2=110.20
f_0_3=110.24
f_0=np.array([f_0_1,f_0_2,f_0_3])
f_0_mean=f_0.mean()
sig_f_0=np.sqrt(((f_0-f_0_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_0_mean =',f_0_mean, ', sig_f_0 =', sig_f_0

f_1_1=123.85   #f_1_1=123.80 eigentlich
f_1_2=123.79
f_1_3=123.74    #f_1_1=123.79 eigentlich
f_1=np.array([f_1_1,f_1_2,f_1_3])
f_1_mean=f_1.mean()
sig_f_1=np.sqrt(((f_1-f_1_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_1_mean =',f_1_mean, ', sig_f_1 =', sig_f_1

f_2_1=138.98
f_2_2=139.04
f_2_3=138.83
f_2=np.array([f_2_1,f_2_2,f_2_3])
f_2_mean=f_2.mean()
sig_f_2=np.sqrt(((f_2-f_2_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_2_mean =',f_2_mean, ', sig_f_2 =', sig_f_2

f_3_1=156.03
f_3_2=155.69
f_3_3=155.96
f_3=np.array([f_3_1,f_3_2,f_3_3])
f_3_mean=f_3.mean()
sig_f_3=np.sqrt(((f_3-f_3_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_3_mean =',f_3_mean, ', sig_f_3 =', sig_f_3

f_4_1=174.60
f_4_2=174.31
f_4_3=174.46
f_4=np.array([f_4_1,f_4_2,f_4_3])
f_4_mean=f_4.mean()
sig_f_4=np.sqrt(((f_4-f_4_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_4_mean =',f_4_mean, ', sig_f_4 =', sig_f_4

f_5_1=194.42
f_5_2=196.00
f_5_3=195.97
f_5=np.array([f_5_1,f_5_2,f_5_3])
f_5_mean=f_5.mean()
sig_f_5=np.sqrt(((f_5-f_5_mean).sum())**2/(6)) #n=3 => n*(n-1)=6
print 'f_5_mean =',f_5_mean, ', sig_f_5 =', sig_f_5

f_ges=np.array([f_0_mean,f_1_mean,f_2_mean,f_3_mean,f_4_mean,f_5_mean])
sig_f_ges=np.array([sig_f_0,sig_f_1,sig_f_2,sig_f_3,sig_f_4,sig_f_5])

'Längen'
l_0=64.9*10**(-2)
l_1=58.0*10**(-2) #vorher 54.6 
l_2=51.5*10**(-2)
l_3=45.9*10**(-2)
l_4=40.9*10**(-2)
l_5=36.4*10**(-2)

l_ges=np.array([l_0,l_1,l_2,l_3,l_4,l_5])
sig_l=1*10**(-3) #m
#eins_durch_l=1/l_ges
#sig_eins_durch_l=sig_l/(l_ges**2)

'2.Wert Fehlerhaft'
l_ges=np.delete(l_ges,1)
eins_durch_l=1/l_ges
sig_eins_durch_l=sig_l/(l_ges**2)

f_ges=np.delete(f_ges,1)
sig_f_ges=np.delete(sig_f_ges,1)



#%%
'lineare Regression'

data=p.lineare_regression_xy(eins_durch_l,f_ges,sig_eins_durch_l,sig_f_ges)
print data
m=data[0]
print "m=", m
sig_m=data[1]
print 'Fehler auf m=',sig_m
print "Relativer Fehler=", sig_m/m

chi2_n=data[4]/3
print chi2_n

plot_A='\n'+'A='+str(round(data[0],3))+'+/-'+str(round(data[1],3))+'m/s'
plot_B='\n'+'B='+str(round(data[2],3))+'+/-'+str(round(data[3],3))+'Hz'
plot_chi='\n'+'chi^2/f='+str(round(chi2_n,3))

#x=np.linspace(1.4,2.8)
plt.errorbar(eins_durch_l,f_ges,sig_eins_durch_l,sig_f_ges,fmt='.')
plt.plot(eins_durch_l,eins_durch_l*data[0]+data[2],label='f=A/l+B'+plot_A+plot_B+plot_chi)
plt.ylim(105,200)
plt.xlabel('1/l in 1/m',fontsize='large')
plt.ylabel('f in Hz',fontsize='large')
plt.title('Lineare Regression')
plt.legend(loc=2)
plt.show()

#%%
x1=[1,2,3,4,5]
'Residuen'
res=f_ges-eins_durch_l*data[0]-data[2]
'Fehler aufs Residuum'
sigf=np.sqrt(np.square(x1)*data[1]**2+data[3]**2)
sigres=np.sqrt(np.square(sig_f_ges)+np.square(sigf))
#print res

plt.errorbar(x1,res,yerr=sigres,fmt='.')
plt.hlines(0,-1,6)
plt.xlim(-1,6)
plt.xlabel('Messwert')
plt.ylabel('Residuum in Hz')
plt.title('Residuenplot')