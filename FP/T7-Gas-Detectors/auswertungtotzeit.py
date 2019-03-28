# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:08:14 2019

@author: assil
"""
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
#2ms
#1MHz-Generator 10s gemessen
a = [519.5,519.7,519.8]
#1mus
b = [1000014.5,1000014.4,1000014.4]
x1=np.mean(a)
x2=np.mean(b)
s=ufloat(x1,np.std(a)/np.sqrt(3))
l=ufloat(x2,np.std(b)/np.sqrt(3))


print('Totzeit:2 ms')
tms2=1/s
print(tms2)
print('Totzeit:1 mus')
t1mus=1/l
print(t1mus)


#____________________________________________________________________________
#2ms 
#GM-Counter
c=[178.6,179.4,180.9]
#1mus
d=[244.9,250.3,242.8]


x=np.mean(c)

y=np.mean(d)

s=ufloat(x,np.std(c)/np.sqrt(3))
l=ufloat(y,np.std(d)/np.sqrt(3))
print(s)
print(l)
N1=s/(1-s*tms2)
N2=l/(1-l*t1mus)
print(N1)
print(N2)

treal=(1/l)-(1/s)+tms2
print(treal)
Countswithoutoffset=[0.0, 0.4, 3.3, 9.3, 14.7, 22.2, 30.9, 43.2, 53.1, 66.3, 69.6, 83.9, 94.1, 104.1, 114.6, 120.8, 136.6, 143.7, 154.0, 161.2, 168.1, 176.1, 186.6, 191.5, 200.3, 206.5, 210.4, 216.4, 221.4, 226.4, 230.7, 236.0, 237.4, 243.2, 245.7, 249.2, 256.7, 254.4, 253.3, 259.9, 261.4, 263.7, 268.6, 263.9, 271.9, 267.8]

errCounts=[0.0, 0.8944271909999159, 1.9235384061671346, 3.1144823004794873, 3.8858718455450894, 4.753945729601885, 5.594640292279746, 6.603029607687671, 7.314369419163897, 8.167006795638168, 8.366600265340756, 9.181503144910424, 9.72111104761179, 10.222524150130436, 10.723805294763608, 11.009087155618309, 11.704699910719626, 12.004165943538101, 12.425779653607254, 12.712198865656562, 12.98075498574717, 13.285330255586423, 13.674794331177344, 13.852797551397336, 14.166862743741113, 14.384018909887459, 14.51895313030523, 14.724129855444769, 14.892951352905172, 15.05988047761336, 15.201973556088038, 15.375304875026057, 15.42076522096099, 15.607690412101336, 15.687574701017363, 15.798734126505199, 16.034338152851834, 15.962455951387932, 15.927962832703999, 16.133815419794537, 16.180234856144704, 16.2511538051918, 16.401219466856727, 16.25730604989646, 16.501515081955354, 16.376812876747415]
cc=unp.uarray(Countswithoutoffset,errCounts)

a=[]
i=0
while i<len(cc ):
    N=cc[i]/(1-cc[i]*0.000422)
    a.append(N)
    i+=1
    
print(a)
print(unp.nominal_values(a))
print(unp.std_devs(a))

# Characterization of Deathtime:
''' First of all we generate a Plus of 1 MHz with the Generator
that is linked to the variable totzeitstufe(2ms,1mus) and write the counts down(counts over 10s measured 3 times).
With N1=s/(1-s*tms2)=N2=l/(1-l*t1mus)=N we can corrtect the Stufe if we consider that the smaller stufe is correct 
with a measured countrate of 10**6 for 1mus and 519.67+-0.07 for 2ms we get a corrected stufe of 1.92ms.

now we can calculate with the same method the auflösungszeit of the GM-Counter.
we consider that the real stufe is between 1mus and 2ms . 
with treal=(1/l)-(1/s)+tms2 , s=179.6+-0.6 and l=246.0+-1.8 we got a auflösungszeit treal of 422+-35 ms

stever-Diagram
with a stever we  can estimate the auflösungszeit on the oscilliscope out of the dead time an the relax time that we can read from the scope 

the deadtime is the distance between the main peak and the next incoming peak, during that time the GM-Counter can not detect anything.
 We estimate the dead time with 224+/-4 mus (for that we read the dead time out of the steverdiagram 3 time with a min and max value and mean over all)
 The realaxtime is the distance bewteen main peak and the next main peak , therfore we got 505+/-11mus (same method as before)


correction GM-Characteristic

with the calculated auflösungszeit we can now correct the Characteistic GM-Curve for the detected count with N=n/(1-n*treal) ,the error calculation is gaussian  propagation of uncertainty 


'''