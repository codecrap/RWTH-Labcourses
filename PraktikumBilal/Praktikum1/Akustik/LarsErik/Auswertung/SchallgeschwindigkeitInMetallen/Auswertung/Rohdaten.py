# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np


"Rohdaten Schallgeschwindigkeit in Metallen"
"Fehler:"

sig_m=0.0001
sig_d=0.01*10**(-3)
sig_l=1*10**(-3)

"1. Stange"
m_1=1.3019

d_1_1=12.47*10**(-3)
d_1_2=12.47*10**(-3)
d_1_3=12.47*10**(-3)
d_1_4=12.47*10**(-3)
d_1_5=12.47*10**(-3)

l_1= 1.299 
d_1=np.array([d_1_1,d_1_2,d_1_3,d_1_4,d_1_5])
d_1_mean=d_1.mean()
#sig_d_1=np.sqrt(((d_1-d_1_mean).sum())**2/(20))
sig_d_1=np.sqrt((np.square(d_1-d_1_mean)).sum()/20) #n=5 => n*(n-1)=20
#print sig_d_1
"2. Stange"
m_2=1.3249

d_2_1=12.01*10**(-3)
d_2_2=12.00*10**(-3)
d_2_3=11.99*10**(-3)
d_2_4=11.99*10**(-3)
d_2_5=12.01*10**(-3)

l_2= 1.50 
d_2=np.array([d_2_1,d_2_2,d_2_3,d_2_4,d_2_5])
d_2_mean=d_2.mean()
sig_d_2=np.sqrt((np.square(d_2-d_2_mean)).sum()/20) #n=5 => n*(n-1)=20
#print sig_d_2

'3. Stange (Vermutlich Aluminium)'
m_3 = 1157.0 * 10**(-3) #kg
l_3 = 130.1 * 10**(-2) #m

d_3_1=11.97 * 10**(-3) #m
d_3_2=11.97 * 10**(-3) #m
d_3_3=11.95 * 10**(-3) #m
d_3_4=11.97 * 10**(-3) #m
d_3_5=11.96 * 10**(-3) #m

d_3=np.array([d_3_1,d_3_2,d_3_3,d_3_4,d_3_5])
d_3_mean=d_3.mean()
sig_d_3=np.sqrt((np.square(d_3-d_3_mean)).sum()/20) #n=5 => n*(n-1)=20
#print sig_d_3

'4. Stange (vermutlich Messing)'
m_4 = 1236.4 * 10**(-3) #kg
l_4 = 129.9 * 10**(-2) #m

d_4_1=11.97 * 10**(-3) #m
d_4_2=11.97 * 10**(-3) #m
d_4_3=12.00 * 10**(-3) #m
d_4_4=11.97 * 10**(-3) #m
d_4_5=11.98 * 10**(-3) #m

d_4=np.array([d_4_1,d_4_2,d_4_3,d_4_4,d_4_5])
d_4_mean=d_4.mean()
sig_d_4=np.sqrt((np.square(d_4-d_4_mean)).sum()/20) #n=5 => n*(n-1)=20
#print sig_d_4
#print "means",d_1_mean,d_2_mean,d_3_mean,d_4_mean
"Frequenz 1"
f_1_1=1511.52
f_1_2=1511.54
f_1_3=1511.52
f_1_4=1511.53
f_1_5=1511.51
f_1_6=1511.53
f_1_7=1511.54
f_1_8=1511.51
f_1_9=1511.51
f_1_10=1511.48

"Frequenz 2"
f_2_1=1728.17
f_2_2=1728.18
f_2_3=1728.18
f_2_4=1728.19
f_2_5=1728.18
f_2_6=1728.17
f_2_7=1728.18
f_2_8=1728.20
f_2_9=1728.19
f_2_10=1728.20

"Frequenz 3"
f_3_1=1884.03
f_3_2=1884.04
f_3_3=1884.07
f_3_4=1884.06
f_3_5=1884.09
f_3_6=1884.11
f_3_7=1884.13
f_3_8=1884.14
f_3_9=1884.13
f_3_10=1884.14 

"Frequenz 4"
f_4_1=1348.48
f_4_2=1348.50
f_4_3=1348.49
f_4_4=1348.50
f_4_5=1348.50
f_4_6=1348.54
f_4_7=1348.55
f_4_8=1348.55
f_4_9=1348.52
f_4_10=1348.52


f_1=np.array([f_1_1,f_1_2,f_1_3,f_1_4,f_1_5,f_1_6,f_1_7,f_1_8,f_1_9,f_1_10])
f_1_mean=f_1.mean()
sig_f_1=np.sqrt((np.square(f_1-f_1_mean)).sum()/90) #n=10 => n*(n-1)=90

f_2=np.array([f_2_1,f_2_2,f_2_3,f_2_4,f_2_5,f_2_6,f_2_7,f_2_8,f_2_9,f_2_10])
f_2_mean=f_2.mean()
sig_f_2=np.sqrt((np.square(f_2-f_2_mean)).sum()/90) #n=10 => n*(n-1)=90

f_3=np.array([f_3_1,f_3_2,f_3_3,f_3_4,f_3_5,f_3_6,f_3_7,f_3_8,f_3_9,f_3_10])
f_3_mean=f_3.mean()
sig_f_3=np.sqrt((np.square(f_3-f_3_mean)).sum()/90) #n=10 => n*(n-1)=90

f_4=np.array([f_4_1,f_4_2,f_4_3,f_4_4,f_4_5,f_4_6,f_4_7,f_4_8,f_4_9,f_4_10])
f_4_mean=f_4.mean()
sig_f_4=np.sqrt((np.square(f_4-f_4_mean)).sum()/90) #n=10 => n*(n-1)=90
print "mittelwert f",f_1_mean,f_2_mean,f_3_mean,f_4_mean
print "Fehler f", sig_f_1, sig_f_2, sig_f_3, sig_f_4